#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace triposplat {

struct decoder_point_set {
    // Token-major normalized coordinates in [0, 1), xyz interleaved.
    std::vector<float> points;
    std::vector<float> log_probs;
    size_t size() const { return points.size() / 3; }
};

// Receives token-major xyz and the current octree resolution. Returns eight
// logits per input point, also token-major. The e2e runner implements this by
// executing build_octree_decoder on Vulkan and copying only its small logits
// result to the host.
using octree_logits_fn = std::function<std::vector<float>(
    const std::vector<float> & points, int resolution)>;

// Matches OctreeProbabilityFixedlenDecoder.sample(..., algo="systematic") for
// the single-item inference batch used by TripoSplat.
decoder_point_set sample_octree(
    const octree_logits_fn & infer,
    size_t num_points,
    int max_level = 8,
    float temperature = 1.0f,
    uint64_t seed = 42);

struct gaussian_cloud {
    // All fields are Gaussian-major. rotation is the raw quaternion delta;
    // opacity and scaling have already had their model activations applied.
    std::vector<float> xyz;       // [N,3], world coordinates in [-.5,.5]
    std::vector<float> features;  // [N,3], SH degree-zero coefficients
    std::vector<float> opacity;   // [N]
    std::vector<float> opacity_logit; // [N], before sigmoid (for finite PLY export)
    std::vector<float> scaling;   // [N,3]
    std::vector<float> rotation;  // [N,4], before adding identity bias
    size_t size() const { return opacity.size(); }
};

struct gaussian_config {
    size_t gaussians_per_point = 32;
    float perturb_size = 1.5f;
    float offset_scale = 0.05f;
    float minimum_kernel_size = 0.0009f;
    float scaling_bias = 0.004f;
    float opacity_bias = 0.1f;
    float rotation_lr = 0.1f;
    bool perturb_offset = true;
    bool learned_offset_scale = true;
};

// features is point-major [num_points, 480] for the production configuration.
// This is build_gs_decoder's output converted from ggml's channel-major tensor.
gaussian_cloud build_gaussians(
    const decoder_point_set & points,
    const std::vector<float> & features,
    const gaussian_config & config = {});

std::vector<uint8_t> gaussian_to_ply(const gaussian_cloud & cloud);
std::vector<uint8_t> gaussian_to_splat(const gaussian_cloud & cloud);
void save_binary_file(const std::string & path, const std::vector<uint8_t> & data);

} // namespace triposplat
