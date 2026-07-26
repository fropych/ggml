#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace triposplat {

struct voxel_conversion_options {
    std::string input_ply;
    std::string output_path;
    uint32_t resolution = 64;
    float iso = 11.345f;
    float opacity_threshold = 0.10f;
    float tolerance = 0.125f;
    uint32_t integration_steps = 10;
    float color_weight_power = 0.625f;
    int vulkan_device = 0;
};

struct voxel_conversion_result {
    std::string device_name;
    uint64_t gaussian_count = 0;
    uint64_t aabb_candidate_pairs = 0;
    uint64_t occupied_voxels = 0;
    uint64_t output_bytes = 0;
    uint64_t converter_gpu_bytes = 0;
    double setup_milliseconds = 0.0;
    double conversion_milliseconds = 0.0;
    double gpu_milliseconds = 0.0;
};

// Convert a binary little-endian Gaussian-splat PLY to the dependency-free
// TSVOXEL v2 occupancy-bitset/RGB8 format using Vulkan compute.
voxel_conversion_result convert_gaussian_ply_to_voxels(
    const voxel_conversion_options & options);

} // namespace triposplat
