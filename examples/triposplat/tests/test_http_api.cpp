#include "server.h"

#include "civetweb.h"
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct contract_options {
    fs::path binary;
    fs::path input_image;
    fs::path model_directory;
    fs::path work_directory;
    int device = 0;
    uint16_t port = 0;
};

struct http_response {
    int status = 0;
    std::string body;
};

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::string option_value(int & index, int argc, char ** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(
            std::string("missing value after ") + argv[index - 1]);
    }
    return argv[index];
}

long long strict_integer(
    const std::string & value, const char * name) {
    size_t consumed = 0;
    const long long result = std::stoll(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + name);
    }
    return result;
}

contract_options parse_options(int argc, char ** argv) {
    contract_options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--binary") {
            options.binary = option_value(index, argc, argv);
        } else if (option == "--input") {
            options.input_image = option_value(index, argc, argv);
        } else if (option == "--model-dir") {
            options.model_directory = option_value(index, argc, argv);
        } else if (option == "--work-dir") {
            options.work_directory = option_value(index, argc, argv);
        } else if (option == "--device") {
            const long long value = strict_integer(
                option_value(index, argc, argv), "--device");
            if (value < std::numeric_limits<int>::min() ||
                value > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("--device is out of range");
            }
            options.device = int(value);
        } else if (option == "--port") {
            const long long value = strict_integer(
                option_value(index, argc, argv), "--port");
            if (value < 1 || value > 65535) {
                throw std::invalid_argument("--port must be in 1..65535");
            }
            options.port = uint16_t(value);
        } else if (option == "--help") {
            std::cout
                << "Usage: " << argv[0]
                << " --binary FILE --input IMAGE --model-dir DIR"
                   " [--device N] [--port N] [--work-dir DIR]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument(
                "unknown test option: " + option);
        }
    }
    if (options.binary.empty() ||
        options.input_image.empty() ||
        options.model_directory.empty()) {
        throw std::invalid_argument(
            "--binary, --input, and --model-dir are required");
    }
    options.binary = fs::absolute(options.binary);
    options.input_image = fs::absolute(options.input_image);
    options.model_directory = fs::absolute(options.model_directory);
    check(fs::is_regular_file(options.binary), "CLI binary does not exist");
    check(
        fs::is_regular_file(options.input_image),
        "input image does not exist");
    check(
        fs::is_directory(options.model_directory),
        "model directory does not exist");
    check(
        fs::is_regular_file(
            options.binary.parent_path() /
            "assets/flow_positions.safetensors"),
        "runtime asset is not beside the CLI binary");
    if (options.port == 0) {
#ifdef _WIN32
        const uint32_t process_id = GetCurrentProcessId();
#else
        const uint32_t process_id = uint32_t(getpid());
#endif
        options.port = uint16_t(20'000 + process_id % 20'000);
    }
    return options;
}

struct working_tree {
    fs::path path;
    bool owned = false;
    bool keep = false;

    explicit working_tree(const fs::path & requested) {
        if (!requested.empty()) {
            path = fs::absolute(requested);
            fs::create_directories(path);
            return;
        }
        const auto stamp =
            std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() /
            ("triposplat-http-contract-" + std::to_string(stamp));
        fs::create_directories(path);
        owned = true;
    }

    ~working_tree() {
        if (!owned || keep) return;
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

#ifdef _WIN32
std::string windows_quote(const std::string & value) {
    std::string quoted = "\"";
    size_t backslashes = 0;
    for (const char character : value) {
        if (character == '\\') {
            ++backslashes;
        } else if (character == '"') {
            quoted.append(backslashes * 2 + 1, '\\');
            quoted.push_back('"');
            backslashes = 0;
        } else {
            quoted.append(backslashes, '\\');
            backslashes = 0;
            quoted.push_back(character);
        }
    }
    quoted.append(backslashes * 2, '\\');
    quoted.push_back('"');
    return quoted;
}
#endif

void run_process(const std::vector<std::string> & arguments) {
    check(!arguments.empty(), "empty subprocess command");
#ifdef _WIN32
    std::string command;
    for (const std::string & argument : arguments) {
        if (!command.empty()) command.push_back(' ');
        command += windows_quote(argument);
    }
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    std::vector<char> mutable_command(command.begin(), command.end());
    mutable_command.push_back('\0');
    if (!CreateProcessA(
            nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
            0, nullptr, nullptr, &startup, &process)) {
        throw std::runtime_error("CreateProcess failed");
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    check(code == 0, "CLI subprocess failed with exit code " +
        std::to_string(code));
#else
    const pid_t child = fork();
    if (child < 0) throw std::runtime_error("fork failed");
    if (child == 0) {
        std::vector<char *> values;
        values.reserve(arguments.size() + 1);
        for (const std::string & argument : arguments) {
            values.push_back(const_cast<char *>(argument.c_str()));
        }
        values.push_back(nullptr);
        execv(values.front(), values.data());
        _exit(127);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("waitpid failed");
    }
    check(
        WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "CLI subprocess failed");
#endif
}

std::string read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open file: " + path.string());
    }
    return {
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
}

void compare_file(
    const fs::path & reference,
    const std::string & actual,
    const std::string & label) {
    const std::string expected = read_file(reference);
    if (expected == actual) return;
    size_t mismatch = 0;
    while (mismatch < expected.size() &&
           mismatch < actual.size() &&
           expected[mismatch] == actual[mismatch]) {
        ++mismatch;
    }
    throw std::runtime_error(
        label + " differs from CLI reference at byte " +
        std::to_string(mismatch) + " (CLI " +
        std::to_string(expected.size()) + " bytes, HTTP " +
        std::to_string(actual.size()) + " bytes)");
}

http_response request(
    uint16_t port,
    const std::string & method,
    const std::string & path,
    const std::string & content_type = {},
    const std::string & body = {}) {
    char error[512]{};
    mg_connection * connection = mg_connect_client(
        "127.0.0.1", port, 0, error, sizeof(error));
    if (!connection) {
        throw std::runtime_error(
            "HTTP connection failed: " + std::string(error));
    }
    struct connection_guard {
        mg_connection * value;
        ~connection_guard() {
            if (value) mg_close_connection(value);
        }
    } guard{connection};

    mg_printf(
        connection,
        "%s %s HTTP/1.1\r\n"
        "Host: 127.0.0.1:%u\r\n"
        "Connection: close\r\n"
        "Content-Length: %zu\r\n",
        method.c_str(), path.c_str(), unsigned(port), body.size());
    if (!content_type.empty()) {
        mg_printf(
            connection, "Content-Type: %s\r\n",
            content_type.c_str());
    }
    mg_printf(connection, "\r\n");
    size_t offset = 0;
    while (offset < body.size()) {
        const int written = mg_write(
            connection, body.data() + offset, body.size() - offset);
        if (written <= 0) {
            throw std::runtime_error("failed to write HTTP request body");
        }
        offset += size_t(written);
    }

    if (mg_get_response(connection, error, sizeof(error), 900'000) < 0) {
        throw std::runtime_error(
            "HTTP response failed: " + std::string(error));
    }
    const mg_response_info * info =
        mg_get_response_info(connection);
    check(info != nullptr, "HTTP response has no metadata");

    http_response response;
    response.status = info->status_code;
    std::vector<char> buffer(64 * 1024);
    for (;;) {
        const int count =
            mg_read(connection, buffer.data(), buffer.size());
        if (count < 0) {
            throw std::runtime_error("failed to read HTTP response");
        }
        if (count == 0) break;
        response.body.append(buffer.data(), size_t(count));
    }
    if (info->content_length >= 0) {
        check(
            uint64_t(info->content_length) == response.body.size(),
            "HTTP response Content-Length mismatch");
    }
    return response;
}

json json_request(
    uint16_t port,
    const std::string & method,
    const std::string & path,
    int expected_status,
    const json & body = nullptr) {
    const bool has_body = !body.is_null();
    const http_response response = request(
        port, method, path,
        has_body ? "application/json" : "",
        has_body ? body.dump() : "");
    check(
        response.status == expected_status,
        method + " " + path + " returned " +
        std::to_string(response.status) + ", expected " +
        std::to_string(expected_status) + ": " + response.body);
    return response.body.empty()
        ? json(nullptr)
        : json::parse(response.body);
}

json upload_artifact(
    uint16_t port,
    const fs::path & path,
    const std::string & type) {
    const std::string boundary =
        "TriposplatContractBoundary7f5e8a1b";
    std::string body;
    body += "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"type\"\r\n\r\n";
    body += type + "\r\n";
    body += "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"" +
        path.filename().string() + "\"\r\n";
    body += "Content-Type: application/octet-stream\r\n\r\n";
    body += read_file(path);
    body += "\r\n--" + boundary + "--\r\n";
    const http_response response = request(
        port, "POST", "/v1/artifacts",
        "multipart/form-data; boundary=" + boundary, body);
    check(
        response.status == 201,
        "artifact upload failed: " + response.body);
    return json::parse(response.body);
}

json wait_for_job(uint16_t port, const std::string & id) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::minutes(15);
    for (;;) {
        const json job = json_request(
            port, "GET", "/v1/jobs/" + id, 200);
        const std::string status = job.at("status").get<std::string>();
        if (status == "succeeded") return job;
        if (status == "failed" ||
            status == "cancelled" ||
            status == "expired") {
            throw std::runtime_error(
                "job " + id + " ended as " + status + ": " +
                job.dump());
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("timed out waiting for job " + id);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::string download_artifact(
    uint16_t port, const std::string & id) {
    const http_response response = request(
        port, "GET", "/v1/artifacts/" + id + "/content");
    check(
        response.status == 200,
        "artifact download failed: " + response.body);
    return response.body;
}

std::string enqueue_job(
    uint16_t port,
    const std::string & path,
    const json & body) {
    const json response =
        json_request(port, "POST", path, 202, body);
    check(
        response.at("status") == "queued",
        "new job was not queued");
    return response.at("job_id").get<std::string>();
}

void run_contract(
    const contract_options & options,
    const fs::path & work) {
    const fs::path cli_prefix = work / "cli-generation";
    constexpr int steps = 2;
    constexpr int num_gaussians = 1024;
    constexpr uint64_t seed = 42;

    run_process({
        options.binary.string(),
        "generate",
        options.input_image.string(),
        "--model-dir", options.model_directory.string(),
        "--output", cli_prefix.string(),
        "--steps", std::to_string(steps),
        "--guidance", "3",
        "--num-gaussians", std::to_string(num_gaussians),
        "--seed", std::to_string(seed),
        "--erode-radius", "1",
        "--device", std::to_string(options.device),
    });
    const fs::path cli_ply = cli_prefix.string() + ".ply";
    const fs::path cli_splat = cli_prefix.string() + ".splat";
    check(fs::is_regular_file(cli_ply), "CLI did not create PLY");
    check(fs::is_regular_file(cli_splat), "CLI did not create SPLAT");

    for (const int resolution : {32, 64}) {
        run_process({
            options.binary.string(),
            "voxelize",
            cli_ply.string(),
            "--output",
            (work / ("cli-" + std::to_string(resolution) + ".tsvox")).string(),
            "--resolution", std::to_string(resolution),
            "--device", std::to_string(options.device),
        });
    }

    triposplat::server_config config;
    config.host = "127.0.0.1";
    config.port = options.port;
    config.artifacts.directory = work / "artifacts";
    config.artifacts.ttl = std::chrono::hours(1);
    config.artifacts.max_artifacts = 32;
    config.artifacts.max_bytes = 2ull * 1024ull * 1024ull * 1024ull;
    config.cleanup_interval = std::chrono::seconds(1);
    config.vulkan_device = options.device;
    config.model_directory = options.model_directory.string();
    config.runtime_asset_directory =
        (options.binary.parent_path() / "assets").string();

    triposplat::rest_server server(config);
    server.start();
    check(server.running(), "REST server did not start");
    check(server.port() == options.port, "REST server bound wrong port");

    const json health =
        json_request(options.port, "GET", "/health", 200);
    check(health.at("status") == "ok", "health check is not ready");
    const json devices =
        json_request(options.port, "GET", "/v1/devices", 200);
    check(
        devices.at("devices").is_array() &&
        !devices.at("devices").empty(),
        "device list is empty");

    json_request(
        options.port, "POST", "/v1/generations", 400,
        {{"input_artifact_id", "missing"}, {"unknown", 1}});
    json_request(options.port, "GET", "/missing", 404);

    const json input = upload_artifact(
        options.port, options.input_image, "input_image");
    const std::string input_id = input.at("id").get<std::string>();
    check(input.at("type") == "input_image", "uploaded artifact type changed");
    compare_file(
        options.input_image, download_artifact(options.port, input_id),
        "uploaded input image");

    const std::string generation_job = enqueue_job(
        options.port, "/v1/generations", {
            {"input_artifact_id", input_id},
            {"seed", seed},
            {"steps", steps},
            {"guidance", 3.0},
            {"num_gaussians", num_gaussians},
            {"erode_radius", 1},
        });
    const std::string cancelled_job = enqueue_job(
        options.port, "/v1/generations", {
            {"input_artifact_id", input_id},
            {"seed", seed + 1},
            {"steps", 1},
            {"guidance", 3.0},
            {"num_gaussians", 32},
            {"erode_radius", 1},
        });
    json_request(
        options.port, "DELETE", "/v1/artifacts/" + input_id, 409);
    const json cancelled = json_request(
        options.port, "DELETE",
        "/v1/jobs/" + cancelled_job, 200);
    check(
        cancelled.at("status") == "cancelled",
        "queued generation job was not cancelled");

    const json generation =
        wait_for_job(options.port, generation_job);
    const std::string ply_id =
        generation.at("artifacts").at("gaussian_ply").get<std::string>();
    const std::string splat_id =
        generation.at("artifacts").at("splat").get<std::string>();
    compare_file(
        cli_ply, download_artifact(options.port, ply_id),
        "generated PLY");
    compare_file(
        cli_splat, download_artifact(options.port, splat_id),
        "generated SPLAT");

    const json uploaded_ply = upload_artifact(
        options.port, cli_ply, "gaussian_ply");
    const std::string uploaded_ply_id =
        uploaded_ply.at("id").get<std::string>();
    check(
        uploaded_ply.at("type") == "gaussian_ply",
        "uploaded PLY artifact type changed");
    compare_file(
        cli_ply, download_artifact(options.port, uploaded_ply_id),
        "uploaded Gaussian PLY");
    json_request(
        options.port, "DELETE",
        "/v1/artifacts/" + uploaded_ply_id, 204);

    json_request(
        options.port, "POST", "/v1/generations", 400,
        {{"input_artifact_id", ply_id}});
    json_request(
        options.port, "POST", "/v1/voxelizations", 400, {
            {"input_artifact_id", ply_id},
            {"resolution", 32},
            {"chunk_depth", 33},
        });

    std::vector<std::pair<std::string, std::string>> voxel_jobs;
    for (const int resolution : {32, 64}) {
        const std::string id = enqueue_job(
            options.port, "/v1/voxelizations", {
                {"input_artifact_id", ply_id},
                {"resolution", resolution},
            });
        const json voxel = wait_for_job(options.port, id);
        const std::string artifact_id =
            voxel.at("artifacts").at("output").get<std::string>();
        compare_file(
            work / ("cli-" + std::to_string(resolution) + ".tsvox"),
            download_artifact(options.port, artifact_id),
            "voxel " + std::to_string(resolution));
        voxel_jobs.emplace_back(id, artifact_id);
    }

    for (const auto & item : voxel_jobs) {
        json_request(
            options.port, "DELETE",
            "/v1/artifacts/" + item.second, 204);
        const json expired = json_request(
            options.port, "GET", "/v1/jobs/" + item.first, 200);
        check(
            expired.at("status") == "expired",
            "deleted voxel output did not expire its job");
    }
    json_request(
        options.port, "DELETE", "/v1/artifacts/" + splat_id, 204);
    const json expired_generation = json_request(
        options.port, "GET", "/v1/jobs/" + generation_job, 200);
    check(
        expired_generation.at("status") == "expired",
        "deleted generation output did not expire its job");

    json_request(
        options.port, "DELETE", "/v1/artifacts/" + ply_id, 204);
    json_request(
        options.port, "DELETE", "/v1/artifacts/" + input_id, 204);
    const http_response missing = request(
        options.port, "GET",
        "/v1/artifacts/" + input_id + "/content");
    check(missing.status == 404, "deleted artifact remained downloadable");

    server.stop();
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const contract_options options = parse_options(argc, argv);
        working_tree work(options.work_directory);
        try {
            std::cout << "HTTP contract work directory: "
                      << work.path << '\n';
            run_contract(options, work.path);
            std::cout
                << "HTTP generation and voxelization contract passed\n";
            return 0;
        } catch (...) {
            work.keep = true;
            std::cerr << "preserving failed contract output in "
                      << work.path << '\n';
            throw;
        }
    } catch (const std::exception & error) {
        std::cerr << "HTTP contract failed: " << error.what() << '\n';
        return 1;
    }
}
