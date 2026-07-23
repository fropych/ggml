#pragma once

#include "model-store.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace triposplat {

struct generate_options {
    std::string input_image;
    std::string output_prefix = "triposplat";
    std::string asset_directory;
    model_store_config models;
    uint64_t seed = 42;
    int steps = 20;
    float guidance = 3.0f;
    size_t num_gaussians = 32768;
    int erode_radius = 1;
    bool keep_temporary = false;
};

struct generate_result {
    std::string ply_path;
    std::string splat_path;
    double elapsed_seconds = 0.0;
};

class pipeline {
public:
    explicit pipeline(int vulkan_device = 0);
    ~pipeline();
    pipeline(const pipeline &) = delete;
    pipeline & operator=(const pipeline &) = delete;

    const std::string & device_description() const;
    generate_result generate(const generate_options & options);

private:
    struct implementation;
    std::unique_ptr<implementation> impl_;
};

} // namespace triposplat
