#pragma once

#include "artifact-store.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace triposplat {

struct server_config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    artifact_store_config artifacts;
    std::chrono::seconds cleanup_interval =
        std::chrono::seconds(60);
    int vulkan_device = 0;
    std::string model_directory;
    std::string runtime_asset_directory;
};

class rest_server {
public:
    explicit rest_server(server_config config);
    ~rest_server();

    rest_server(const rest_server &) = delete;
    rest_server & operator=(const rest_server &) = delete;

    void start();
    void stop();
    bool running() const;
    uint16_t port() const;

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

int run_server_command(
    int argc,
    char ** argv,
    const std::filesystem::path & executable_directory);

} // namespace triposplat
