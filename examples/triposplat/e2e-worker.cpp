#include "e2e-worker.h"

#include "biref-model.h"
#include "decoder-model.h"
#include "decoder-runtime.h"
#include "dino-model.h"
#include "flow-model.h"
#include "host-tensor-io.h"
#include "safetensors.h"
#include "vae-model.h"

#include "ggml.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace triposplat {
namespace {

struct graph_execution {
    ggml_backend_t backend = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;

    graph_execution(ggml_backend_t backend_, ggml_context * ctx,
                    const std::vector<ggml_tensor *> & outputs, size_t capacity)
        : backend(backend_) {
        graph = ggml_new_graph_custom(ctx, capacity, false);
        for (ggml_tensor * output : outputs) ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                throw std::runtime_error(std::string("Vulkan does not support e2e op ") +
                                         ggml_op_desc(node));
            }
        }
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!allocator || !ggml_gallocr_alloc_graph(allocator, graph)) {
            throw std::runtime_error("Vulkan e2e graph allocation failed");
        }
    }

    ~graph_execution() {
        release();
    }

    void release() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
    }

    void compute(const char * label) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string(label) + " Vulkan execution failed");
        }
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("%s: %.3f ms\n", label, ms);
    }
};

host_dtype tensor_host_type(const ggml_tensor * tensor) {
    switch (tensor->type) {
        case GGML_TYPE_F16:  return host_dtype::f16;
        case GGML_TYPE_BF16: return host_dtype::bf16;
        case GGML_TYPE_F32:  return host_dtype::f32;
        case GGML_TYPE_I32:  return host_dtype::i32;
        case GGML_TYPE_I64:  return host_dtype::i64;
        default: throw std::runtime_error("cannot export unsupported ggml tensor type");
    }
}

host_tensor export_tensor(const std::string & name, ggml_tensor * tensor,
                          std::vector<int64_t> shape) {
    host_tensor result = allocate_host_tensor(name, tensor_host_type(tensor),
                                              std::move(shape));
    if (result.bytes.size() != ggml_nbytes(tensor)) {
        throw std::runtime_error("non-contiguous e2e output tensor: " + name);
    }
    ggml_backend_tensor_get(tensor, result.bytes.data(), 0, result.bytes.size());
    return result;
}

std::vector<float> tensor_f32(ggml_tensor * tensor) {
    const size_t count = size_t(ggml_nelements(tensor));
    std::vector<float> result(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, result.data(), 0, count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(values[0]));
        for (size_t i = 0; i < count; ++i) result[i] = ggml_fp16_to_fp32(values[i]);
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> values(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(values[0]));
        for (size_t i = 0; i < count; ++i) result[i] = ggml_bf16_to_fp32(values[i]);
    } else {
        throw std::runtime_error("e2e tensor must be floating point");
    }
    return result;
}

void upload_f32(ggml_tensor * tensor, const std::vector<float> & values) {
    if (size_t(ggml_nelements(tensor)) != values.size()) {
        throw std::runtime_error("e2e upload element count mismatch");
    }
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> converted(values.size());
        for (size_t i = 0; i < values.size(); ++i) converted[i] = ggml_fp32_to_fp16(values[i]);
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(converted[0]));
    } else if (tensor->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> converted(values.size());
        for (size_t i = 0; i < values.size(); ++i) converted[i] = ggml_fp32_to_bf16(values[i]);
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(converted[0]));
    } else {
        throw std::runtime_error("e2e upload tensor must be floating point");
    }
}

ggml_context * make_context(std::vector<uint8_t> & metadata, size_t bytes) {
    metadata.resize(bytes);
    ggml_init_params params { metadata.size(), metadata.data(), true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("failed to allocate e2e graph metadata");
    return ctx;
}

int run_biref(int argc, char ** argv, ggml_backend_t backend) {
    if (argc != 5) throw std::runtime_error("--run-biref MODEL INPUT OUTPUT");
    weight_store weights(backend, argv[2], weight_policy::f16);
    weight_store inputs(backend, argv[3], weight_policy::native);
    std::vector<uint8_t> metadata;
    ggml_context * ctx = make_context(metadata, 256u * 1024u * 1024u);
    biref_inputs in { inputs.get("pixels"),
        { inputs.get("full_mask0"), inputs.get("full_mask1"),
          inputs.get("full_mask2"), inputs.get("full_mask3") },
        { inputs.get("half_mask0"), inputs.get("half_mask1"),
          inputs.get("half_mask2"), inputs.get("half_mask3") } };
    ggml_tensor * output = build_birefnet(ctx, weights, in);
    graph_execution execution(backend, ctx, {output}, 32768);
    execution.compute("BiRefNet e2e");
    save_safetensors(argv[4], {export_tensor(
        "alpha", output, {output->ne[3], output->ne[2], output->ne[1], output->ne[0]})});
    execution.release();
    ggml_free(ctx);
    return 0;
}

int run_dino(int argc, char ** argv, ggml_backend_t backend) {
    if (argc != 5) throw std::runtime_error("--run-dino MODEL INPUT OUTPUT");
    weight_store weights(backend, argv[2], weight_policy::f16);
    weight_store inputs(backend, argv[3], weight_policy::native);
    std::vector<uint8_t> metadata;
    ggml_context * ctx = make_context(metadata, 64u * 1024u * 1024u);
    dino_inputs in { inputs.get("pixels"), inputs.get("rope_cos"), inputs.get("rope_sin") };
    ggml_tensor * output = build_dino_model(ctx, weights, in, 32);
    graph_execution execution(backend, ctx, {output}, 8192);
    execution.compute("DINO e2e");
    save_safetensors(argv[4], {export_tensor(
        "feature1", output, {output->ne[2], output->ne[1], output->ne[0]})});
    execution.release();
    ggml_free(ctx);
    return 0;
}

int run_vae(int argc, char ** argv, ggml_backend_t backend) {
    if (argc != 5) throw std::runtime_error("--run-vae MODEL INPUT OUTPUT");
    weight_store weights(backend, argv[2], weight_policy::f16);
    weight_store inputs(backend, argv[3], weight_policy::native);
    std::vector<uint8_t> metadata;
    ggml_context * ctx = make_context(metadata, 64u * 1024u * 1024u);
    ggml_tensor * output = build_vae_encoder_stochastic(
        ctx, weights, inputs.get("image"), inputs.get("noise"));
    graph_execution execution(backend, ctx, {output}, 8192);
    execution.compute("VAE e2e");
    save_safetensors(argv[4], {export_tensor(
        "feature2", output, {output->ne[2], output->ne[1], output->ne[0]})});
    execution.release();
    ggml_free(ctx);
    return 0;
}

int run_flow(int argc, char ** argv, ggml_backend_t backend) {
    if (argc < 5 || argc > 7) {
        throw std::runtime_error("--run-flow MODEL INPUT OUTPUT [STEPS] [GUIDANCE]");
    }
    const int steps = argc > 5 ? std::max(1, std::stoi(argv[5])) : 20;
    const float guidance = argc > 6 ? std::stof(argv[6]) : 3.0f;
    weight_store weights(backend, argv[2], weight_policy::native);
    weight_store inputs(backend, argv[3], weight_policy::native);
    std::vector<uint8_t> metadata;
    ggml_context * ctx = make_context(metadata, 96u * 1024u * 1024u);
    flow_inputs in { inputs.get("latent"), inputs.get("feature1"), inputs.get("feature2"),
                     inputs.get("camera"), inputs.get("timestep"), inputs.get("positions"),
                     inputs.get("position_freqs") };
    flow_outputs outputs = build_flow_outputs(ctx, weights, in, 24);
    graph_execution execution(backend, ctx, {outputs.latent, outputs.camera}, 8192);
    std::vector<float> latent = tensor_f32(in.latent);
    std::vector<float> camera = tensor_f32(in.camera);
    const std::vector<float> condition1 = tensor_f32(in.feature1);
    const std::vector<float> condition2 = tensor_f32(in.feature2);
    const std::vector<float> zero1(condition1.size()), zero2(condition2.size());
    const float shift = 3.0f;
    for (int step = 0; step < steps; ++step) {
        const float u = float(step) / steps, up = float(step + 1) / steps;
        const float base = 1.0f - u, basep = 1.0f - up;
        const float t = shift * base / (1.0f + (shift - 1.0f) * base);
        const float tp = shift * basep / (1.0f + (shift - 1.0f) * basep);
        const float timestep = 1000.0f * t;
        ggml_backend_tensor_set(in.timestep, &timestep, 0, sizeof(timestep));
        upload_f32(in.latent, latent); upload_f32(in.camera, camera);
        upload_f32(in.feature1, condition1); upload_f32(in.feature2, condition2);
        execution.compute("Flow conditional");
        const std::vector<float> positive_latent = tensor_f32(outputs.latent);
        const std::vector<float> positive_camera = tensor_f32(outputs.camera);
        upload_f32(in.feature1, zero1); upload_f32(in.feature2, zero2);
        execution.compute("Flow unconditional");
        const std::vector<float> negative_latent = tensor_f32(outputs.latent);
        const std::vector<float> negative_camera = tensor_f32(outputs.camera);
        const float dt = t - tp;
        for (size_t i = 0; i < latent.size(); ++i) {
            latent[i] -= (guidance * positive_latent[i] -
                          (guidance - 1.0f) * negative_latent[i]) * dt;
        }
        for (size_t i = 0; i < camera.size(); ++i) {
            camera[i] -= (guidance * positive_camera[i] -
                          (guidance - 1.0f) * negative_camera[i]) * dt;
        }
        std::printf("Flow sampler step %d/%d\n", step + 1, steps);
    }
    save_safetensors(argv[4], {
        copy_host_tensor("latent", host_dtype::f32,
                         {in.latent->ne[2], in.latent->ne[1], in.latent->ne[0]}, latent),
        copy_host_tensor("camera", host_dtype::f32,
                         {in.camera->ne[2], in.camera->ne[1], in.camera->ne[0]}, camera),
    });
    execution.release();
    ggml_free(ctx);
    return 0;
}

int run_decode(int argc, char ** argv, ggml_backend_t backend) {
    if (argc < 5 || argc > 7) {
        throw std::runtime_error("--run-decode MODEL INPUT OUTPUT_PREFIX [GAUSSIANS] [SEED]");
    }
    const size_t num_gaussians = argc > 5 ? size_t(std::stoull(argv[5])) : 32768;
    const uint64_t seed = argc > 6 ? uint64_t(std::stoull(argv[6])) : 42;
    if (num_gaussians < 32 || num_gaussians % 32 != 0) {
        throw std::runtime_error("Gaussian count must be a positive multiple of 32");
    }
    weight_store weights(backend, argv[2], weight_policy::native);
    weight_store inputs(backend, argv[3], weight_policy::native);
    ggml_tensor * condition = inputs.get("condition");
    ggml_tensor * frequencies = inputs.get("position_freqs");
    auto infer = [&](const std::vector<float> & points, int resolution) {
        const int64_t count = int64_t(points.size() / 3);
        std::vector<uint8_t> metadata;
        ggml_context * ctx = make_context(metadata, 64u * 1024u * 1024u);
        ggml_tensor * point_tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, count, 1);
        ggml_tensor * level_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        octree_inputs in {point_tensor, level_tensor, condition, frequencies};
        ggml_tensor * output = build_octree_decoder(ctx, weights, in, 4);
        graph_execution execution(backend, ctx, {output}, 8192);
        ggml_backend_tensor_set(point_tensor, points.data(), 0, points.size() * sizeof(float));
        const float level = float(resolution);
        ggml_backend_tensor_set(level_tensor, &level, 0, sizeof(level));
        execution.compute("Octree level");
        std::vector<float> result = tensor_f32(output);
        execution.release();
        ggml_free(ctx);
        return result;
    };
    decoder_point_set points = sample_octree(infer, num_gaussians / 32, 8, 1.0f, seed);

    std::vector<uint8_t> metadata;
    ggml_context * ctx = make_context(metadata, 64u * 1024u * 1024u);
    ggml_tensor * point_tensor = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, 3, int64_t(points.size()), 1);
    gs_inputs in {point_tensor, condition, frequencies};
    ggml_tensor * output = build_gs_decoder(ctx, weights, in, 16);
    graph_execution execution(backend, ctx, {output}, 8192);
    ggml_backend_tensor_set(point_tensor, points.points.data(), 0,
                            points.points.size() * sizeof(float));
    execution.compute("Gaussian decoder e2e");
    gaussian_cloud cloud = build_gaussians(points, tensor_f32(output));
    const std::string prefix = argv[4];
    save_binary_file(prefix + ".ply", gaussian_to_ply(cloud));
    save_binary_file(prefix + ".splat", gaussian_to_splat(cloud));
    std::printf("wrote %zu Gaussians to %s.{ply,splat}\n", cloud.size(), prefix.c_str());
    execution.release();
    ggml_free(ctx);
    return 0;
}

} // namespace

bool is_e2e_worker_mode(const std::string & mode) {
    return mode == "--run-biref" || mode == "--run-dino" ||
           mode == "--run-vae" || mode == "--run-flow" ||
           mode == "--run-decode";
}

int run_e2e_worker(const std::string & mode, int argc, char ** argv,
                   ggml_backend_t backend) {
    if (mode == "--run-biref") return run_biref(argc, argv, backend);
    if (mode == "--run-dino") return run_dino(argc, argv, backend);
    if (mode == "--run-vae") return run_vae(argc, argv, backend);
    if (mode == "--run-flow") return run_flow(argc, argv, backend);
    if (mode == "--run-decode") return run_decode(argc, argv, backend);
    throw std::runtime_error("unknown e2e worker mode: " + mode);
}

} // namespace triposplat
