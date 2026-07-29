#include "server.h"

#include "ggml-vulkan.h"
#include "job-manager.h"
#include "model-store.h"

#include "civetweb.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace triposplat {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

std::once_flag civetweb_initialization;
volatile std::sig_atomic_t signal_stop_requested = 0;

void initialize_civetweb() {
    std::call_once(civetweb_initialization, [] {
        if (mg_init_library(MG_FEATURES_DEFAULT) !=
            MG_FEATURES_DEFAULT) {
            throw std::runtime_error("failed to initialize CivetWeb");
        }
    });
}

void signal_handler(int) {
    signal_stop_requested = 1;
}

void send_json(
    mg_connection * connection,
    int status,
    const json & value) {
    const std::string body = value.dump();
    mg_printf(
        connection,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        mg_get_response_code_text(connection, status),
        body.size());
    if (!body.empty()) mg_write(connection, body.data(), body.size());
}

int send_error(
    mg_connection * connection,
    int status,
    const std::string & code,
    const std::string & message) {
    send_json(connection, status, {
        {"error", {
            {"code", code},
            {"message", message},
        }},
    });
    return status;
}

std::string method(const mg_request_info * request) {
    return request && request->request_method
        ? request->request_method
        : "";
}

std::string request_path(const mg_request_info * request) {
    if (!request) return {};
    const char * path = request->local_uri
        ? request->local_uri
        : request->request_uri;
    return path ? path : "";
}

bool starts_with(const std::string & value, const std::string & prefix) {
    return value.size() >= prefix.size() &&
        value.compare(0, prefix.size(), prefix) == 0;
}

std::optional<std::string> route_id(
    const std::string & path,
    const std::string & prefix,
    const std::string & suffix = {}) {
    if (!starts_with(path, prefix)) return std::nullopt;
    if (!suffix.empty()) {
        if (path.size() <= prefix.size() + suffix.size() ||
            path.compare(
                path.size() - suffix.size(),
                suffix.size(), suffix) != 0) {
            return std::nullopt;
        }
    }
    const size_t end = path.size() - suffix.size();
    const std::string id =
        path.substr(prefix.size(), end - prefix.size());
    if (id.empty() || id.find('/') != std::string::npos) {
        return std::nullopt;
    }
    return id;
}

std::string safe_download_name(const artifact_record & record) {
    std::string name = record.original_name.empty()
        ? record.file_name
        : fs::path(record.original_name).filename().string();
    for (char & character : name) {
        if (character == '"' || character == '\r' ||
            character == '\n' || character == '\\') {
            character = '_';
        }
    }
    return name.empty() ? "artifact.bin" : name;
}

const char * mime_type(const artifact_record & record) {
    if (record.type == "input_image") {
        return "application/octet-stream";
    }
    if (record.type == "gaussian_ply") {
        return "application/octet-stream";
    }
    if (record.type == "splat") {
        return "application/octet-stream";
    }
    if (record.type == "tsvox") {
        return "application/octet-stream";
    }
    if (record.type == "log") return "text/plain";
    return "application/octet-stream";
}

json artifact_json(const artifact_record & record) {
    return {
        {"id", record.id},
        {"type", record.type},
        {"state", record.state},
        {"size_bytes", record.size_bytes},
        {"created_at", artifact_store::format_time(record.created_at)},
        {"expires_at", artifact_store::format_time(record.expires_at)},
        {"original_name", record.original_name},
        {"producer_job_id", record.producer_job_id.empty()
            ? json(nullptr) : json(record.producer_job_id)},
        {"parent_artifact_id", record.parent_artifact_id.empty()
            ? json(nullptr) : json(record.parent_artifact_id)},
        {"content_url", "/v1/artifacts/" + record.id + "/content"},
    };
}

json job_json(const job_snapshot & job) {
    json artifacts = json::object();
    if (!job.ply_artifact_id.empty()) {
        artifacts["gaussian_ply"] = job.ply_artifact_id;
    }
    if (!job.splat_artifact_id.empty()) {
        artifacts["splat"] = job.splat_artifact_id;
    }
    if (!job.output_artifact_id.empty()) {
        artifacts["output"] = job.output_artifact_id;
    }

    json metrics = json::object();
    if (job.type == "generation" &&
        (job.status == job_status::succeeded ||
         job.status == job_status::expired)) {
        metrics = {
            {"elapsed_seconds",
             job.generation_metrics.elapsed_seconds},
        };
    } else if (job.type == "voxelization" &&
               (job.status == job_status::succeeded ||
                job.status == job_status::expired)) {
        const voxel_conversion_result & value = job.voxel_metrics;
        metrics = {
            {"device_name", value.device_name},
            {"gaussian_count", value.gaussian_count},
            {"aabb_candidate_pairs", value.aabb_candidate_pairs},
            {"occupied_voxels", value.occupied_voxels},
            {"output_bytes", value.output_bytes},
            {"converter_gpu_bytes", value.converter_gpu_bytes},
            {"converter_device_bytes", value.converter_device_bytes},
            {"chunk_count", value.chunk_count},
            {"max_chunk_voxels", value.max_chunk_voxels},
            {"setup_milliseconds", value.setup_milliseconds},
            {"conversion_milliseconds",
             value.conversion_milliseconds},
            {"gpu_milliseconds", value.gpu_milliseconds},
        };
    }

    json result = {
        {"id", job.id},
        {"type", job.type},
        {"status", job_manager::status_name(job.status)},
        {"input_artifact_id", job.input_artifact_id},
        {"created_at", artifact_store::format_time(job.created_at)},
        {"started_at", job.started_at
            ? json(artifact_store::format_time(*job.started_at))
            : json(nullptr)},
        {"finished_at", job.finished_at
            ? json(artifact_store::format_time(*job.finished_at))
            : json(nullptr)},
        {"artifacts", std::move(artifacts)},
        {"metrics", std::move(metrics)},
    };
    result["error"] =
        job.error.empty() ? json(nullptr) : json(job.error);
    return result;
}

std::string read_json_body(
    mg_connection * connection,
    size_t maximum_bytes) {
    const mg_request_info * request =
        mg_get_request_info(connection);
    if (!request || request->content_length < 0) {
        throw std::invalid_argument(
            "Content-Length is required");
    }
    if (uint64_t(request->content_length) > maximum_bytes) {
        throw std::length_error("request body is too large");
    }
    const char * content_type = mg_get_header(
        connection, "Content-Type");
    if (!content_type ||
        !starts_with(content_type, "application/json")) {
        throw std::invalid_argument(
            "Content-Type must be application/json");
    }
    std::string body(
        size_t(request->content_length), '\0');
    size_t offset = 0;
    while (offset < body.size()) {
        const int count = mg_read(
            connection, body.data() + offset,
            body.size() - offset);
        if (count <= 0) {
            throw std::runtime_error(
                "request body ended before Content-Length");
        }
        offset += size_t(count);
    }
    return body;
}

void reject_unknown_fields(
    const json & value,
    std::initializer_list<const char *> allowed) {
    if (!value.is_object()) {
        throw std::invalid_argument("request body must be a JSON object");
    }
    std::unordered_set<std::string> names;
    for (const char * item : allowed) names.insert(item);
    for (auto item = value.begin(); item != value.end(); ++item) {
        if (names.count(item.key()) == 0) {
            throw std::invalid_argument(
                "unknown JSON field: " + item.key());
        }
    }
}

std::string required_string(
    const json & value, const char * name) {
    if (!value.contains(name) ||
        !value.at(name).is_string()) {
        throw std::invalid_argument(
            std::string(name) + " must be a string");
    }
    const std::string result =
        value.at(name).get<std::string>();
    if (result.empty()) {
        throw std::invalid_argument(
            std::string(name) + " must not be empty");
    }
    return result;
}

template <typename T>
T optional_number(
    const json & value, const char * name, T fallback) {
    if (!value.contains(name)) return fallback;
    try {
        return value.at(name).get<T>();
    } catch (const json::exception &) {
        throw std::invalid_argument(
            std::string(name) + " has an invalid numeric value");
    }
}

struct upload_state {
    artifact_store * artifacts = nullptr;
    fs::path temporary_path;
    std::string temporary_path_string;
    std::string original_name;
    std::string type;
    bool file_seen = false;
    bool file_stored = false;
    bool invalid = false;
};

int upload_field_found(
    const char * key,
    const char * filename,
    char * path,
    size_t path_length,
    void * user_data) {
    auto & state = *static_cast<upload_state *>(user_data);
    const std::string name = key ? key : "";
    const bool has_filename = filename && filename[0] != '\0';
    if (name == "file" && has_filename && !state.file_seen) {
        state.file_seen = true;
        state.original_name = fs::path(filename).filename().string();
        state.temporary_path =
            state.artifacts->make_temporary_file(".upload");
        state.temporary_path_string =
            state.temporary_path.string();
        if (state.temporary_path_string.size() + 1 > path_length) {
            state.invalid = true;
            return MG_FORM_FIELD_STORAGE_ABORT;
        }
        std::memcpy(
            path, state.temporary_path_string.c_str(),
            state.temporary_path_string.size() + 1);
        return MG_FORM_FIELD_STORAGE_STORE;
    }
    if (name == "type" && !has_filename) {
        return MG_FORM_FIELD_STORAGE_GET;
    }
    if (name == "file") state.invalid = true;
    return name == "file"
        ? MG_FORM_FIELD_STORAGE_ABORT
        : MG_FORM_FIELD_STORAGE_SKIP;
}

int upload_field_get(
    const char * key,
    const char * value,
    size_t value_length,
    void * user_data) {
    auto & state = *static_cast<upload_state *>(user_data);
    if (key && std::string(key) == "type") {
        if (state.type.size() + value_length > 64) {
            state.invalid = true;
            return MG_FORM_FIELD_HANDLE_ABORT;
        }
        state.type.append(value, value_length);
        return MG_FORM_FIELD_HANDLE_GET;
    }
    return MG_FORM_FIELD_HANDLE_NEXT;
}

int upload_field_store(
    const char *,
    long long,
    void * user_data) {
    auto & state = *static_cast<upload_state *>(user_data);
    state.file_stored = true;
    return MG_FORM_FIELD_HANDLE_NEXT;
}

uint64_t checked_uint64(
    const std::string & value,
    const char * option) {
    size_t consumed = 0;
    const uint64_t parsed = std::stoull(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option);
    }
    return parsed;
}

int checked_int(
    const std::string & value,
    const char * option) {
    size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed);
    if (consumed != value.size() ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option);
    }
    return int(parsed);
}

std::chrono::seconds parse_duration(
    const std::string & value,
    const char * option) {
    if (value.size() < 2) {
        throw std::invalid_argument(
            std::string("invalid duration for ") + option);
    }
    const char suffix = value.back();
    uint64_t multiplier = 0;
    if (suffix == 's') multiplier = 1;
    else if (suffix == 'm') multiplier = 60;
    else if (suffix == 'h') multiplier = 60 * 60;
    else if (suffix == 'd') multiplier = 24 * 60 * 60;
    else {
        throw std::invalid_argument(
            std::string("duration for ") + option +
            " must use s, m, h, or d");
    }
    const uint64_t count = checked_uint64(
        value.substr(0, value.size() - 1), option);
    if (count == 0 ||
        count > uint64_t(std::numeric_limits<int64_t>::max()) /
                    multiplier) {
        throw std::invalid_argument(
            std::string("duration for ") + option +
            " is out of range");
    }
    return std::chrono::seconds(count * multiplier);
}

uint64_t parse_size(
    const std::string & value,
    const char * option) {
    struct suffix {
        const char * text;
        uint64_t multiplier;
    };
    static constexpr std::array<suffix, 4> suffixes = {{
        {"GiB", 1024ull * 1024ull * 1024ull},
        {"MiB", 1024ull * 1024ull},
        {"KiB", 1024ull},
        {"B", 1ull},
    }};
    for (const suffix & item : suffixes) {
        const std::string text = item.text;
        if (value.size() > text.size() &&
            value.compare(
                value.size() - text.size(), text.size(), text) == 0) {
            const uint64_t count = checked_uint64(
                value.substr(0, value.size() - text.size()), option);
            if (count == 0 ||
                count > std::numeric_limits<uint64_t>::max() /
                            item.multiplier) {
                break;
            }
            return count * item.multiplier;
        }
    }
    throw std::invalid_argument(
        std::string("size for ") + option +
        " must use B, KiB, MiB, or GiB");
}

std::string option_value(
    int & index, int argc, char ** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") +
            argv[index - 1]);
    }
    return argv[index];
}

void server_usage(const char * executable) {
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s serve --model-dir DIR [OPTIONS]\n\n"
        "Server options:\n"
        "  --host HOST                 Listen address (default: 127.0.0.1)\n"
        "  --port N                    TCP port, 1..65535 (default: 8080)\n"
        "  --artifact-dir DIR          Artifact store (default: ./artifacts)\n"
        "  --artifact-ttl DURATION     Fixed TTL using s/m/h/d (default: 24h)\n"
        "  --max-artifacts N           Artifact count limit (default: 100)\n"
        "  --max-artifact-bytes SIZE   Store limit using B/KiB/MiB/GiB (default: 50GiB)\n"
        "  --cleanup-interval DURATION Cleanup period (default: 60s)\n"
        "  --device N                  Vulkan device index (default: 0)\n"
        "  --model-dir DIR             TripoSplat model directory (required)\n"
        "  --help                      Show this help\n\n"
        "Runtime assets are loaded from the assets directory beside the executable.\n",
        executable);
}

} // namespace

struct rest_server::implementation {
    server_config config;
    artifact_store artifacts;
    job_manager jobs;
    mg_context * context = nullptr;
    std::atomic<bool> active{false};
    std::mutex housekeeping_mutex;
    std::condition_variable housekeeping_condition;
    std::thread housekeeping;
    uint16_t bound_port = 0;

    explicit implementation(server_config input_config)
        : config(std::move(input_config)),
          artifacts(config.artifacts),
          jobs(artifacts, {
              config.vulkan_device,
              config.model_directory,
              config.runtime_asset_directory}) {
        if (config.host.empty() ||
            config.host.find_first_of(", \t\r\n") != std::string::npos) {
            throw std::invalid_argument("invalid server host");
        }
        if (config.port == 0) {
            throw std::invalid_argument(
                "server port must be in 1..65535");
        }
        if (config.cleanup_interval <=
            std::chrono::seconds::zero()) {
            throw std::invalid_argument(
                "cleanup interval must be positive");
        }
        if (config.model_directory.empty()) {
            throw std::invalid_argument("--model-dir is required");
        }
        resolve_models({config.model_directory});
        const fs::path flow_positions =
            fs::path(config.runtime_asset_directory) /
            "flow_positions.safetensors";
        if (!fs::is_regular_file(flow_positions)) {
            throw std::runtime_error(
                "runtime asset is missing: " +
                flow_positions.string());
        }
        const int count = ggml_backend_vk_get_device_count();
        if (config.vulkan_device < 0 ||
            config.vulkan_device >= count) {
            throw std::invalid_argument(
                "Vulkan device index is out of range");
        }
    }

    ~implementation() {
        stop();
    }

    void start() {
        if (active.load()) {
            throw std::logic_error("REST server is already running");
        }
        initialize_civetweb();
        const std::string listening =
            config.host + ":" + std::to_string(config.port);
        const std::string threads = "8";
        const std::string timeout = "30000";
        const char * options[] = {
            "listening_ports", listening.c_str(),
            "num_threads", threads.c_str(),
            "request_timeout_ms", timeout.c_str(),
            "enable_keep_alive", "yes",
            nullptr,
        };
        mg_callbacks callbacks{};
        context = mg_start(&callbacks, this, options);
        if (!context) {
            throw std::runtime_error(
                "failed to start REST server on " + listening);
        }
        mg_set_request_handler(
            context, "/", &implementation::dispatch, this);
        mg_server_port ports[4]{};
        const int port_count =
            mg_get_server_ports(context, 4, ports);
        if (port_count < 1 || !ports[0].is_bound) {
            mg_stop(context);
            context = nullptr;
            throw std::runtime_error(
                "REST server did not bind a listening port");
        }
        bound_port = uint16_t(ports[0].port);
        active.store(true);
        housekeeping = std::thread([this] {
            std::unique_lock<std::mutex> lock(housekeeping_mutex);
            while (active.load()) {
                if (housekeeping_condition.wait_for(
                        lock, config.cleanup_interval,
                        [this] { return !active.load(); })) {
                    break;
                }
                lock.unlock();
                artifacts.cleanup();
                lock.lock();
            }
        });
    }

    void stop() {
        if (!active.exchange(false)) return;
        housekeeping_condition.notify_all();
        if (context) {
            mg_stop(context);
            context = nullptr;
        }
        if (housekeeping.joinable()) housekeeping.join();
        bound_port = 0;
    }

    static int dispatch(
        mg_connection * connection, void * user_data) {
        auto & self = *static_cast<implementation *>(user_data);
        try {
            return self.handle(connection);
        } catch (const std::length_error & error) {
            return send_error(
                connection, 413, "payload_too_large", error.what());
        } catch (const std::out_of_range & error) {
            return send_error(
                connection, 404, "not_found", error.what());
        } catch (const std::invalid_argument & error) {
            return send_error(
                connection, 400, "invalid_request", error.what());
        } catch (const json::exception & error) {
            return send_error(
                connection, 400, "invalid_json", error.what());
        } catch (const std::exception & error) {
            return send_error(
                connection, 500, "internal_error", error.what());
        }
    }

    int handle(mg_connection * connection) {
        const mg_request_info * request =
            mg_get_request_info(connection);
        const std::string verb = method(request);
        const std::string path = request_path(request);

        if (path == "/health") {
            if (verb != "GET") return method_not_allowed(connection, "GET");
            send_json(connection, 200, {
                {"status", "ok"},
                {"service", "triposplat-vulkan"},
                {"api_version", "v1"},
            });
            return 200;
        }
        if (path == "/v1/devices") {
            if (verb != "GET") return method_not_allowed(connection, "GET");
            return devices(connection);
        }
        if (path == "/v1/artifacts") {
            if (verb != "POST") {
                return method_not_allowed(connection, "POST");
            }
            return upload_artifact(connection);
        }
        if (path == "/v1/generations") {
            if (verb != "POST") {
                return method_not_allowed(connection, "POST");
            }
            return enqueue_generation(connection);
        }
        if (path == "/v1/voxelizations") {
            if (verb != "POST") {
                return method_not_allowed(connection, "POST");
            }
            return enqueue_voxelization(connection);
        }

        if (const auto id = route_id(
                path, "/v1/artifacts/", "/content")) {
            if (verb != "GET") {
                return method_not_allowed(connection, "GET");
            }
            return download_artifact(connection, *id);
        }
        if (const auto id = route_id(path, "/v1/artifacts/")) {
            if (verb == "GET") return get_artifact(connection, *id);
            if (verb == "DELETE") {
                return delete_artifact(connection, *id);
            }
            return method_not_allowed(connection, "GET, DELETE");
        }
        if (const auto id = route_id(path, "/v1/jobs/")) {
            if (verb == "GET") return get_job(connection, *id);
            if (verb == "DELETE") return cancel_job(connection, *id);
            return method_not_allowed(connection, "GET, DELETE");
        }
        return send_error(
            connection, 404, "not_found", "route not found");
    }

    int method_not_allowed(
        mg_connection * connection, const char * allowed) {
        const json body = {
            {"error", {
                {"code", "method_not_allowed"},
                {"message", "HTTP method is not allowed for this route"},
            }},
        };
        const std::string text = body.dump();
        mg_printf(
            connection,
            "HTTP/1.1 405 Method Not Allowed\r\n"
            "Allow: %s\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n\r\n",
            allowed, text.size());
        mg_write(connection, text.data(), text.size());
        return 405;
    }

    int devices(mg_connection * connection) {
        json devices = json::array();
        const int count = ggml_backend_vk_get_device_count();
        for (int index = 0; index < count; ++index) {
            char description[256]{};
            ggml_backend_vk_get_device_description(
                index, description, sizeof(description));
            devices.push_back({
                {"index", index},
                {"name", description},
                {"selected", index == config.vulkan_device},
            });
        }
        send_json(connection, 200, {{"devices", std::move(devices)}});
        return 200;
    }

    int upload_artifact(mg_connection * connection) {
        const mg_request_info * request =
            mg_get_request_info(connection);
        constexpr uint64_t multipart_overhead = 1024ull * 1024ull;
        const uint64_t maximum_request =
            config.artifacts.max_bytes >
                    uint64_t(std::numeric_limits<long long>::max()) -
                        multipart_overhead
                ? uint64_t(std::numeric_limits<long long>::max())
                : config.artifacts.max_bytes + multipart_overhead;
        if (!request || request->content_length < 0) {
            throw std::invalid_argument(
                "Content-Length is required");
        }
        if (uint64_t(request->content_length) > maximum_request) {
            throw std::length_error("upload is too large");
        }
        const char * content_type =
            mg_get_header(connection, "Content-Type");
        if (!content_type ||
            !starts_with(content_type, "multipart/form-data")) {
            throw std::invalid_argument(
                "Content-Type must be multipart/form-data");
        }

        upload_state state;
        state.artifacts = &artifacts;
        mg_form_data_handler handler{};
        handler.field_found = upload_field_found;
        handler.field_get = upload_field_get;
        handler.field_store = upload_field_store;
        handler.user_data = &state;

        try {
            const int result =
                mg_handle_form_request(connection, &handler);
            if (result < 0 || state.invalid ||
                !state.file_seen || !state.file_stored ||
                !artifact_store::valid_type(state.type) ||
                (state.type != "input_image" &&
                 state.type != "gaussian_ply")) {
                throw std::invalid_argument(
                    "multipart form requires one file and type=input_image|gaussian_ply");
            }
            const artifact_record record = artifacts.commit_file(
                state.temporary_path, state.type,
                state.original_name);
            send_json(connection, 201, artifact_json(record));
            return 201;
        } catch (...) {
            if (!state.temporary_path.empty()) {
                std::error_code ignored;
                fs::remove(state.temporary_path, ignored);
            }
            throw;
        }
    }

    int get_artifact(
        mg_connection * connection, const std::string & id) {
        const auto record = artifacts.get(id);
        if (!record) {
            return send_error(
                connection, 404, "artifact_not_found",
                "artifact does not exist or has expired");
        }
        send_json(connection, 200, artifact_json(*record));
        return 200;
    }

    int download_artifact(
        mg_connection * connection, const std::string & id) {
        artifact_lease lease = artifacts.acquire(id);
        if (!lease) {
            return send_error(
                connection, 404, "artifact_not_found",
                "artifact does not exist or has expired");
        }
        const artifact_record & record = lease.record();
        const std::string name = safe_download_name(record);
        mg_printf(
            connection,
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: %s\r\n"
            "Content-Length: %llu\r\n"
            "Content-Disposition: attachment; filename=\"%s\"\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n",
            mime_type(record),
            static_cast<unsigned long long>(record.size_bytes),
            name.c_str());
        std::ifstream input(record.data_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "failed to open artifact content");
        }
        std::array<char, 64 * 1024> buffer{};
        while (input) {
            input.read(buffer.data(), buffer.size());
            const std::streamsize count = input.gcount();
            if (count > 0 &&
                mg_write(
                    connection, buffer.data(),
                    size_t(count)) != count) {
                break;
            }
        }
        return 200;
    }

    int delete_artifact(
        mg_connection * connection, const std::string & id) {
        switch (artifacts.remove(id)) {
            case artifact_remove_result::removed:
                mg_printf(
                    connection,
                    "HTTP/1.1 204 No Content\r\n"
                    "Content-Length: 0\r\n"
                    "Connection: close\r\n\r\n");
                return 204;
            case artifact_remove_result::not_found:
                return send_error(
                    connection, 404, "artifact_not_found",
                    "artifact does not exist");
            case artifact_remove_result::in_use:
                return send_error(
                    connection, 409, "artifact_in_use",
                    "artifact is used by an active request or job");
        }
        return 500;
    }

    int enqueue_generation(mg_connection * connection) {
        const json body = json::parse(read_json_body(connection, 64 * 1024));
        reject_unknown_fields(body, {
            "input_artifact_id", "seed", "steps", "guidance",
            "num_gaussians", "erode_radius"});
        generation_request request;
        request.input_artifact_id =
            required_string(body, "input_artifact_id");
        request.seed = optional_number<uint64_t>(
            body, "seed", request.seed);
        request.steps = optional_number<int>(
            body, "steps", request.steps);
        request.guidance = optional_number<float>(
            body, "guidance", request.guidance);
        const uint64_t gaussians = optional_number<uint64_t>(
            body, "num_gaussians", request.num_gaussians);
        if (gaussians > std::numeric_limits<size_t>::max()) {
            throw std::invalid_argument(
                "num_gaussians is out of range");
        }
        request.num_gaussians = size_t(gaussians);
        request.erode_radius = optional_number<int>(
            body, "erode_radius", request.erode_radius);
        const std::string id = jobs.enqueue_generation(request);
        send_json(connection, 202, {
            {"job_id", id},
            {"status", "queued"},
            {"status_url", "/v1/jobs/" + id},
        });
        return 202;
    }

    int enqueue_voxelization(mg_connection * connection) {
        const json body = json::parse(read_json_body(connection, 64 * 1024));
        reject_unknown_fields(body, {
            "input_artifact_id", "resolution", "opacity_threshold",
            "color_weight_power", "iso", "tolerance",
            "integration_steps", "chunk_depth"});
        voxelization_request request;
        request.input_artifact_id =
            required_string(body, "input_artifact_id");
        request.resolution = optional_number<uint32_t>(
            body, "resolution", request.resolution);
        request.opacity_threshold = optional_number<float>(
            body, "opacity_threshold", request.opacity_threshold);
        request.color_weight_power = optional_number<float>(
            body, "color_weight_power", request.color_weight_power);
        request.iso = optional_number<float>(
            body, "iso", request.iso);
        request.tolerance = optional_number<float>(
            body, "tolerance", request.tolerance);
        request.integration_steps = optional_number<uint32_t>(
            body, "integration_steps", request.integration_steps);
        request.chunk_depth = optional_number<uint32_t>(
            body, "chunk_depth", request.chunk_depth);
        const std::string id = jobs.enqueue_voxelization(request);
        send_json(connection, 202, {
            {"job_id", id},
            {"status", "queued"},
            {"status_url", "/v1/jobs/" + id},
        });
        return 202;
    }

    int get_job(
        mg_connection * connection, const std::string & id) {
        const auto job = jobs.get(id);
        if (!job) {
            return send_error(
                connection, 404, "job_not_found",
                "job does not exist");
        }
        send_json(connection, 200, job_json(*job));
        return 200;
    }

    int cancel_job(
        mg_connection * connection, const std::string & id) {
        switch (jobs.cancel(id)) {
            case job_cancel_result::cancelled: {
                const auto job = jobs.get(id);
                send_json(connection, 200, job_json(*job));
                return 200;
            }
            case job_cancel_result::not_found:
                return send_error(
                    connection, 404, "job_not_found",
                    "job does not exist");
            case job_cancel_result::running:
                return send_error(
                    connection, 409, "job_running",
                    "a running GPU job cannot be interrupted safely");
            case job_cancel_result::already_finished:
                return send_error(
                    connection, 409, "job_finished",
                    "job has already finished");
        }
        return 500;
    }
};

rest_server::rest_server(server_config config)
    : impl_(std::make_unique<implementation>(std::move(config))) {}

rest_server::~rest_server() = default;

void rest_server::start() {
    impl_->start();
}

void rest_server::stop() {
    impl_->stop();
}

bool rest_server::running() const {
    return impl_->active.load();
}

uint16_t rest_server::port() const {
    return impl_->bound_port;
}

int run_server_command(
    int argc,
    char ** argv,
    const fs::path & executable_directory) {
    server_config config;
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help") {
            server_usage(argv[0]);
            return 0;
        }
        if (option == "--host") {
            config.host = option_value(index, argc, argv);
        } else if (option == "--port") {
            const uint64_t port = checked_uint64(
                option_value(index, argc, argv), "--port");
            if (port < 1 || port > 65535) {
                throw std::invalid_argument(
                    "--port must be in 1..65535");
            }
            config.port = uint16_t(port);
        } else if (option == "--artifact-dir") {
            config.artifacts.directory =
                option_value(index, argc, argv);
        } else if (option == "--artifact-ttl") {
            config.artifacts.ttl = parse_duration(
                option_value(index, argc, argv), "--artifact-ttl");
        } else if (option == "--max-artifacts") {
            const uint64_t count = checked_uint64(
                option_value(index, argc, argv), "--max-artifacts");
            if (count == 0 ||
                count > std::numeric_limits<size_t>::max()) {
                throw std::invalid_argument(
                    "--max-artifacts is out of range");
            }
            config.artifacts.max_artifacts = size_t(count);
        } else if (option == "--max-artifact-bytes") {
            config.artifacts.max_bytes = parse_size(
                option_value(index, argc, argv),
                "--max-artifact-bytes");
        } else if (option == "--cleanup-interval") {
            config.cleanup_interval = parse_duration(
                option_value(index, argc, argv),
                "--cleanup-interval");
        } else if (option == "--device") {
            config.vulkan_device = checked_int(
                option_value(index, argc, argv), "--device");
        } else if (option == "--model-dir") {
            config.model_directory =
                option_value(index, argc, argv);
        } else {
            throw std::invalid_argument(
                "unknown serve option: " + option);
        }
    }
    if (config.model_directory.empty()) {
        throw std::invalid_argument(
            "serve requires --model-dir");
    }
    config.runtime_asset_directory =
        (executable_directory / "assets").string();

    signal_stop_requested = 0;
    rest_server server(config);
    server.start();
    std::signal(SIGINT, signal_handler);
#ifdef SIGTERM
    std::signal(SIGTERM, signal_handler);
#endif
    std::printf(
        "TripoSplat REST API listening on http://%s:%u\n"
        "Artifacts: %s (TTL %llds, max %zu files / %llu bytes)\n",
        config.host.c_str(),
        unsigned(server.port()),
        fs::absolute(config.artifacts.directory).string().c_str(),
        static_cast<long long>(config.artifacts.ttl.count()),
        config.artifacts.max_artifacts,
        static_cast<unsigned long long>(config.artifacts.max_bytes));
    std::fflush(stdout);

    while (!signal_stop_requested) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    server.stop();
    return 0;
}

} // namespace triposplat
