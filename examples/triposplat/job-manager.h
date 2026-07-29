#pragma once

#include "artifact-store.h"
#include "pipeline.h"
#include "voxel-converter.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

namespace triposplat {

enum class job_status {
    queued,
    running,
    succeeded,
    failed,
    cancelled,
    expired,
};

struct generation_request {
    std::string input_artifact_id;
    uint64_t seed = 42;
    int steps = 20;
    float guidance = 3.0f;
    size_t num_gaussians = 32768;
    int erode_radius = 1;
};

struct voxelization_request {
    std::string input_artifact_id;
    uint32_t resolution = 64;
    float iso = 11.345f;
    float opacity_threshold = 0.10f;
    float tolerance = 0.125f;
    uint32_t integration_steps = 10;
    float color_weight_power = 0.625f;
    uint32_t chunk_depth = 0;
};

struct job_snapshot {
    std::string id;
    std::string type;
    job_status status = job_status::queued;
    std::string error;
    std::string input_artifact_id;
    std::string ply_artifact_id;
    std::string splat_artifact_id;
    std::string output_artifact_id;
    std::chrono::system_clock::time_point created_at;
    std::optional<std::chrono::system_clock::time_point> started_at;
    std::optional<std::chrono::system_clock::time_point> finished_at;
    generate_result generation_metrics;
    voxel_conversion_result voxel_metrics;
};

enum class job_cancel_result {
    cancelled,
    not_found,
    already_finished,
    running,
};

struct job_manager_config {
    int vulkan_device = 0;
    std::string model_directory;
    std::string asset_directory;
};

class job_manager {
public:
    job_manager(
        artifact_store & artifacts,
        job_manager_config config);
    ~job_manager();

    job_manager(const job_manager &) = delete;
    job_manager & operator=(const job_manager &) = delete;

    std::string enqueue_generation(const generation_request & request);
    std::string enqueue_voxelization(
        const voxelization_request & request);

    std::optional<job_snapshot> get(const std::string & id);
    job_cancel_result cancel(const std::string & id);

    static const char * status_name(job_status status);

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace triposplat
