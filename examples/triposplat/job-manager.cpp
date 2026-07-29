#include "job-manager.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace triposplat {
namespace {

namespace fs = std::filesystem;

std::string random_job_id() {
    static std::mutex mutex;
    static std::random_device random;
    static std::mt19937_64 generator(
        (uint64_t(random()) << 32) ^ uint64_t(random()));
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream output;
    output << "job_" << std::hex << std::setfill('0')
           << std::setw(16) << generator()
           << std::setw(16) << generator();
    return output.str();
}

void validate_generation(const generation_request & request) {
    if (request.input_artifact_id.empty()) {
        throw std::invalid_argument(
            "input_artifact_id is required");
    }
    if (request.steps < 1) {
        throw std::invalid_argument("steps must be positive");
    }
    if (!std::isfinite(request.guidance)) {
        throw std::invalid_argument("guidance must be finite");
    }
    if (request.num_gaussians < 32 ||
        request.num_gaussians % 32 != 0) {
        throw std::invalid_argument(
            "num_gaussians must be a positive multiple of 32");
    }
    if (request.erode_radius < 0) {
        throw std::invalid_argument(
            "erode_radius must be non-negative");
    }
}

void validate_voxelization(const voxelization_request & request) {
    if (request.input_artifact_id.empty()) {
        throw std::invalid_argument(
            "input_artifact_id is required");
    }
    if (request.resolution < 2 ||
        request.resolution > 1024 ||
        (request.resolution & (request.resolution - 1)) != 0) {
        throw std::invalid_argument(
            "resolution must be a power of two in 2..1024");
    }
    if (!std::isfinite(request.iso) || request.iso <= 0.0f) {
        throw std::invalid_argument("iso must be positive and finite");
    }
    if (!std::isfinite(request.opacity_threshold) ||
        request.opacity_threshold < 0.0f ||
        request.opacity_threshold > 1.0f) {
        throw std::invalid_argument(
            "opacity_threshold must be finite and in 0..1");
    }
    if (!std::isfinite(request.tolerance) ||
        request.tolerance <= 0.0f) {
        throw std::invalid_argument(
            "tolerance must be positive and finite");
    }
    if (request.integration_steps < 1 ||
        request.integration_steps > 256) {
        throw std::invalid_argument(
            "integration_steps must be in 1..256");
    }
    if (!std::isfinite(request.color_weight_power) ||
        request.color_weight_power <= 0.0f) {
        throw std::invalid_argument(
            "color_weight_power must be positive and finite");
    }
    if (request.chunk_depth > request.resolution) {
        throw std::invalid_argument(
            "chunk_depth must not exceed resolution");
    }
}

struct temporary_directory_guard {
    fs::path path;
    ~temporary_directory_guard() {
        if (!path.empty()) {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    }
};

} // namespace

struct job_manager::implementation {
    struct work_item {
        std::string id;
        std::optional<generation_request> generation;
        std::optional<voxelization_request> voxelization;
        artifact_lease input;
    };

    artifact_store & artifacts;
    job_manager_config config;
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<work_item> queue;
    std::unordered_map<std::string, job_snapshot> jobs;
    bool stopping = false;
    std::thread worker;
    std::unique_ptr<pipeline> generation_pipeline;

    implementation(
        artifact_store & input_artifacts,
        job_manager_config input_config)
        : artifacts(input_artifacts),
          config(std::move(input_config)),
          worker([this] { worker_loop(); }) {}

    ~implementation() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            const auto finished = std::chrono::system_clock::now();
            for (const work_item & item : queue) {
                auto found = jobs.find(item.id);
                if (found != jobs.end() &&
                    found->second.status == job_status::queued) {
                    found->second.status = job_status::cancelled;
                    found->second.error = "server is shutting down";
                    found->second.finished_at = finished;
                }
            }
            queue.clear();
        }
        condition.notify_all();
        if (worker.joinable()) worker.join();
    }

    std::string unique_job_id_locked() {
        for (int attempt = 0; attempt < 100; ++attempt) {
            const std::string id = random_job_id();
            if (jobs.count(id) == 0) return id;
        }
        throw std::runtime_error("failed to allocate a unique job id");
    }

    std::string enqueue(
        std::optional<generation_request> generation,
        std::optional<voxelization_request> voxelization) {
        const std::string input_id = generation
            ? generation->input_artifact_id
            : voxelization->input_artifact_id;
        artifact_lease input = artifacts.acquire(input_id);
        if (!input) {
            throw std::out_of_range(
                "input artifact does not exist or has expired");
        }
        const std::string expected_type =
            generation ? "input_image" : "gaussian_ply";
        if (input.record().type != expected_type) {
            throw std::invalid_argument(
                "input artifact type must be " + expected_type);
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (stopping) throw std::runtime_error("job manager is stopping");
        const std::string id = unique_job_id_locked();
        job_snapshot snapshot;
        snapshot.id = id;
        snapshot.type =
            generation ? "generation" : "voxelization";
        snapshot.status = job_status::queued;
        snapshot.input_artifact_id = input_id;
        snapshot.created_at = std::chrono::system_clock::now();
        jobs.emplace(id, snapshot);
        queue.push_back({
            id, std::move(generation), std::move(voxelization),
            std::move(input)});
        condition.notify_one();
        return id;
    }

    void worker_loop() {
        for (;;) {
            work_item item;
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock, [&] {
                    return stopping || !queue.empty();
                });
                if (stopping) return;
                item = std::move(queue.front());
                queue.pop_front();
                auto found = jobs.find(item.id);
                if (found == jobs.end() ||
                    found->second.status == job_status::cancelled) {
                    continue;
                }
                found->second.status = job_status::running;
                found->second.started_at =
                    std::chrono::system_clock::now();
            }

            try {
                if (item.generation) {
                    run_generation(item, *item.generation);
                } else {
                    run_voxelization(item, *item.voxelization);
                }
                finish(item.id, job_status::succeeded, {});
            } catch (const std::exception & error) {
                finish(item.id, job_status::failed, error.what());
            } catch (...) {
                finish(
                    item.id, job_status::failed,
                    "unknown worker failure");
            }
        }
    }

    void finish(
        const std::string & id,
        job_status status,
        const std::string & error) {
        std::lock_guard<std::mutex> lock(mutex);
        auto found = jobs.find(id);
        if (found == jobs.end()) return;
        found->second.status = status;
        found->second.error = error;
        found->second.finished_at =
            std::chrono::system_clock::now();
    }

    void run_generation(
        const work_item & item,
        const generation_request & request) {
        if (!generation_pipeline) {
            generation_pipeline =
                std::make_unique<pipeline>(config.vulkan_device);
        }
        temporary_directory_guard temporary{
            artifacts.make_temporary_directory(item.id)};
        const fs::path prefix = temporary.path / "result";

        generate_options options;
        options.input_image = item.input.record().data_path.string();
        options.output_prefix = prefix.string();
        options.asset_directory = config.asset_directory;
        options.models.directory = config.model_directory;
        options.seed = request.seed;
        options.steps = request.steps;
        options.guidance = request.guidance;
        options.num_gaussians = request.num_gaussians;
        options.erode_radius = request.erode_radius;
        const generate_result result =
            generation_pipeline->generate(options);

        std::string ply_id;
        std::string splat_id;
        artifact_lease ply_lease;
        try {
            const artifact_record ply = artifacts.commit_file(
                result.ply_path, "gaussian_ply",
                "result.ply", item.id,
                item.input.record().id);
            ply_id = ply.id;
            ply_lease = artifacts.acquire(ply.id);
            if (!ply_lease) {
                throw std::runtime_error(
                    "generated PLY disappeared before job commit");
            }
            const artifact_record splat = artifacts.commit_file(
                result.splat_path, "splat",
                "result.splat", item.id,
                item.input.record().id);
            splat_id = splat.id;
            std::lock_guard<std::mutex> lock(mutex);
            job_snapshot & snapshot = jobs.at(item.id);
            snapshot.ply_artifact_id = ply.id;
            snapshot.splat_artifact_id = splat.id;
            snapshot.generation_metrics = result;
            snapshot.generation_metrics.ply_path.clear();
            snapshot.generation_metrics.splat_path.clear();
        } catch (...) {
            ply_lease = {};
            if (!splat_id.empty()) artifacts.remove(splat_id);
            if (!ply_id.empty()) artifacts.remove(ply_id);
            throw;
        }
    }

    void run_voxelization(
        const work_item & item,
        const voxelization_request & request) {
        temporary_directory_guard temporary{
            artifacts.make_temporary_directory(item.id)};
        const fs::path output = temporary.path / "result.tsvox";
        voxel_conversion_options options;
        options.input_ply = item.input.record().data_path.string();
        options.output_path = output.string();
        options.resolution = request.resolution;
        options.iso = request.iso;
        options.opacity_threshold = request.opacity_threshold;
        options.tolerance = request.tolerance;
        options.integration_steps = request.integration_steps;
        options.color_weight_power = request.color_weight_power;
        options.chunk_depth = request.chunk_depth;
        options.vulkan_device = config.vulkan_device;
        const voxel_conversion_result result =
            convert_gaussian_ply_to_voxels(options);
        std::string output_id;
        try {
            const artifact_record artifact = artifacts.commit_file(
                output, "tsvox", "result.tsvox", item.id,
                item.input.record().id);
            output_id = artifact.id;
            std::lock_guard<std::mutex> lock(mutex);
            job_snapshot & snapshot = jobs.at(item.id);
            snapshot.output_artifact_id = artifact.id;
            snapshot.voxel_metrics = result;
        } catch (...) {
            if (!output_id.empty()) artifacts.remove(output_id);
            throw;
        }
    }

    void mark_expired_outputs(job_snapshot & snapshot) {
        if (snapshot.status != job_status::succeeded) return;
        bool has_output = false;
        bool missing_output = false;
        for (const std::string * id : {
                 &snapshot.ply_artifact_id,
                 &snapshot.splat_artifact_id,
                 &snapshot.output_artifact_id}) {
            if (id->empty()) continue;
            has_output = true;
            if (!artifacts.get(*id)) missing_output = true;
        }
        if (has_output && missing_output) {
            snapshot.status = job_status::expired;
        }
    }
};

job_manager::job_manager(
    artifact_store & artifacts,
    job_manager_config config)
    : impl_(std::make_unique<implementation>(
          artifacts, std::move(config))) {}

job_manager::~job_manager() = default;

std::string job_manager::enqueue_generation(
    const generation_request & request) {
    validate_generation(request);
    if (impl_->config.model_directory.empty()) {
        throw std::runtime_error(
            "server has no model directory configured");
    }
    return impl_->enqueue(request, std::nullopt);
}

std::string job_manager::enqueue_voxelization(
    const voxelization_request & request) {
    validate_voxelization(request);
    return impl_->enqueue(std::nullopt, request);
}

std::optional<job_snapshot> job_manager::get(
    const std::string & id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end()) return std::nullopt;
    job_snapshot snapshot = found->second;
    impl_->mark_expired_outputs(snapshot);
    if (snapshot.status == job_status::expired &&
        found->second.status == job_status::succeeded) {
        found->second.status = job_status::expired;
    }
    return snapshot;
}

job_cancel_result job_manager::cancel(
    const std::string & id) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto found = impl_->jobs.find(id);
    if (found == impl_->jobs.end()) {
        return job_cancel_result::not_found;
    }
    if (found->second.status == job_status::running) {
        return job_cancel_result::running;
    }
    if (found->second.status != job_status::queued) {
        return job_cancel_result::already_finished;
    }
    found->second.status = job_status::cancelled;
    found->second.finished_at = std::chrono::system_clock::now();
    const auto queued = std::find_if(
        impl_->queue.begin(), impl_->queue.end(),
        [&](const implementation::work_item & item) {
            return item.id == id;
        });
    if (queued != impl_->queue.end()) {
        impl_->queue.erase(queued);
    }
    return job_cancel_result::cancelled;
}

const char * job_manager::status_name(job_status status) {
    switch (status) {
        case job_status::queued: return "queued";
        case job_status::running: return "running";
        case job_status::succeeded: return "succeeded";
        case job_status::failed: return "failed";
        case job_status::cancelled: return "cancelled";
        case job_status::expired: return "expired";
    }
    return "unknown";
}

} // namespace triposplat
