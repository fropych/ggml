#include "pipeline.h"

#include "e2e-worker.h"
#include "host-tensor-io.h"
#include "image-io.h"

#include "ggml.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace triposplat {
namespace {

namespace fs = std::filesystem;
constexpr int kCanvas = 1024;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

struct temporary_directory {
    fs::path path;
    bool keep = false;

    explicit temporary_directory(uint64_t seed) {
        const fs::path base = fs::temp_directory_path();
        for (uint64_t attempt = 0; attempt < 100; ++attempt) {
            path = base / ("triposplat-cpp-" + std::to_string(seed) + "-" +
                           std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(path, error)) return;
        }
        throw std::runtime_error("cannot create temporary TripoSplat directory");
    }
    ~temporary_directory() {
        if (!keep) {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
    }
};

float half_to_float(uint16_t value) {
    ggml_fp16_t half;
    std::memcpy(&half, &value, sizeof(value));
    return ggml_fp16_to_fp32(half);
}

float bf16_to_float(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> as_f32(const host_tensor & tensor) {
    const size_t count = tensor.element_count();
    std::vector<float> output(count);
    if (tensor.type == host_dtype::f32) {
        std::memcpy(output.data(), tensor.bytes.data(), tensor.bytes.size());
    } else if (tensor.type == host_dtype::f16 || tensor.type == host_dtype::bf16) {
        const auto * values = reinterpret_cast<const uint16_t *>(tensor.bytes.data());
        for (size_t i = 0; i < count; ++i) output[i] =
            tensor.type == host_dtype::f16 ? half_to_float(values[i]) : bf16_to_float(values[i]);
    } else {
        throw std::runtime_error("pipeline expected a floating-point tensor: " + tensor.name);
    }
    return output;
}

host_tensor f32_tensor(std::string name, std::vector<int64_t> shape,
                       const std::vector<float> & values) {
    size_t expected = 1;
    for (int64_t dimension : shape) expected *= size_t(dimension);
    if (expected != values.size()) {
        throw std::runtime_error("tensor " + name + " has " +
            std::to_string(values.size()) + " values; shape requires " +
            std::to_string(expected));
    }
    return copy_host_tensor(std::move(name), host_dtype::f32, std::move(shape), values);
}

int invoke_worker(ggml_backend_t backend, const std::vector<std::string> & arguments) {
    std::vector<std::string> storage;
    storage.reserve(arguments.size() + 1);
    storage.emplace_back("triposplat-core");
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char *> argv;
    argv.reserve(storage.size());
    for (std::string & value : storage) argv.push_back(value.data());
    const int result = run_e2e_worker(arguments.at(0), int(argv.size()), argv.data(), backend);
    if (result != 0) throw std::runtime_error("Vulkan stage failed: " + arguments.at(0));
    return result;
}

std::vector<float> resize_plane(const std::vector<float> & source, int sw, int sh,
                                int dw, int dh) {
    std::vector<float> output(size_t(dw) * dh);
    const float sx = dw > 1 ? float(sw - 1) / (dw - 1) : 0;
    const float sy = dh > 1 ? float(sh - 1) / (dh - 1) : 0;
    for (int y = 0; y < dh; ++y) {
        const float fy = y * sy; const int y0 = int(fy), y1 = std::min(y0 + 1, sh - 1);
        const float wy = fy - y0;
        for (int x = 0; x < dw; ++x) {
            const float fx = x * sx; const int x0 = int(fx), x1 = std::min(x0 + 1, sw - 1);
            const float wx = fx - x0;
            const float top = source[size_t(y0) * sw + x0] * (1 - wx) +
                              source[size_t(y0) * sw + x1] * wx;
            const float bottom = source[size_t(y1) * sw + x0] * (1 - wx) +
                                 source[size_t(y1) * sw + x1] * wx;
            output[size_t(y) * dw + x] = top * (1 - wy) + bottom * wy;
        }
    }
    return output;
}

std::vector<float> image_chw(const rgba_image & image, bool normalize) {
    std::vector<float> values(size_t(3) * image.width * image.height);
    const size_t plane = size_t(image.width) * image.height;
    for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
        const size_t pixel = size_t(y) * image.width + x;
        for (int c = 0; c < 3; ++c) {
            float value = image.pixels[pixel * 4 + c] / 255.0f;
            if (normalize) value = (value - kMean[c]) / kStd[c];
            values[size_t(c) * plane + pixel] = value;
        }
    }
    return values;
}

host_tensor make_attention_mask(const std::string & name, int height, int width,
                                int window = 12) {
    const int hp = (height + window - 1) / window * window;
    const int wp = (width + window - 1) / window * window;
    const int shift = window / 2;
    std::vector<int> regions(size_t(hp) * wp);
    auto region_axis = [&](int value, int extent) {
        if (value < extent - window) return 0;
        if (value < extent - shift) return 1;
        return 2;
    };
    for (int y = 0; y < hp; ++y) for (int x = 0; x < wp; ++x) {
        regions[size_t(y) * wp + x] = region_axis(y, hp) * 3 + region_axis(x, wp);
    }
    const int windows_y = hp / window, windows_x = wp / window;
    const int count = windows_y * windows_x;
    std::vector<float> mask(size_t(count) * window * window * window * window);
    for (int wy = 0; wy < windows_y; ++wy) for (int wx = 0; wx < windows_x; ++wx) {
        const int index = wy * windows_x + wx;
        for (int a = 0; a < window * window; ++a) {
            const int ay = wy * window + a / window, ax = wx * window + a % window;
            for (int b = 0; b < window * window; ++b) {
                const int by = wy * window + b / window, bx = wx * window + b % window;
                mask[(size_t(index) * 144 + a) * 144 + b] =
                    regions[size_t(ay) * wp + ax] == regions[size_t(by) * wp + bx] ? 0.0f : -100.0f;
            }
        }
    }
    return f32_tensor(name, {count, 144, 144}, mask);
}

std::vector<host_tensor> biref_masks(int size) {
    std::vector<host_tensor> result;
    for (const auto & entry : {std::pair<std::string, int>{"full_", size},
                                {"half_", size / 2}}) {
        int height = (entry.second + 3) / 4, width = height;
        for (int stage = 0; stage < 4; ++stage) {
            result.push_back(make_attention_mask(entry.first + "mask" + std::to_string(stage),
                                                 height, width));
            height = (height + 1) / 2; width = (width + 1) / 2;
        }
    }
    return result;
}

void make_dino_rope(std::vector<float> & cosine, std::vector<float> & sine) {
    constexpr int height = kCanvas / 16, width = kCanvas / 16, dim = 64;
    std::vector<float> inv(dim / 4);
    for (int i = 0; i < dim / 4; ++i) {
        inv[i] = 1.0f / std::pow(100.0f, float(i * 4) / dim);
    }
    cosine.resize(size_t(height) * width * dim);
    sine.resize(cosine.size());
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float coords[2] = {
            2.0f * ((y + 0.5f) / height) - 1.0f,
            2.0f * ((x + 0.5f) / width) - 1.0f,
        };
        size_t base = (size_t(y) * width + x) * dim;
        for (int duplicate = 0; duplicate < 2; ++duplicate) for (int axis = 0; axis < 2; ++axis) {
            for (int i = 0; i < dim / 4; ++i) {
                const int d = duplicate * (dim / 2) + axis * (dim / 4) + i;
                const float angle = 2.0f * float(M_PI) * coords[axis] * inv[i];
                cosine[base + d] = std::cos(angle);
                sine[base + d] = std::sin(angle);
            }
        }
    }
}

std::vector<float> position_freqs_v1() {
    constexpr int count = 1024 / 3 / 2;
    std::vector<float> result(count);
    for (int i = 0; i < count; ++i) {
        const float exponent = i < 16 ? float(i) : float(i - 16) / (count - 16) * 16.0f;
        result[i] = std::pow(2.0f, exponent);
    }
    return result;
}

std::vector<float> position_freqs_v2() {
    constexpr int count = 1024 / 3 / 2;
    std::vector<float> result(count);
    for (int i = 0; i < count; ++i) result[i] =
        std::pow(2.0f, 10.0f * i / float(count - 1));
    return result;
}

rgba_image remove_background(ggml_backend_t backend, rgba_image image,
                             const model_paths & models, const fs::path & work) {
    if (image.has_transparency()) return image;
    rgba_image square = resize_image_bilinear(image, kCanvas, kCanvas);
    std::vector<host_tensor> tensors = biref_masks(kCanvas);
    tensors.push_back(f32_tensor("pixels", {1, 3, kCanvas, kCanvas},
                                 image_chw(square, true)));
    const fs::path input = work / "biref-input.safetensors";
    const fs::path output = work / "biref-output.safetensors";
    save_safetensors(input.string(), tensors);
    invoke_worker(backend, {"--run-biref", models.biref, input.string(), output.string()});
    std::vector<float> alpha = as_f32(load_host_safetensors(output.string()).at("alpha"));
    alpha = resize_plane(alpha, kCanvas, kCanvas, image.width, image.height);
    for (size_t i = 0; i < alpha.size(); ++i) {
        image.pixels[i * 4 + 3] =
            uint8_t(std::clamp(std::lround(alpha[i] * 255.0f), 0l, 255l));
    }
    return image;
}

} // namespace

struct pipeline::implementation {
    ggml_backend_t backend = nullptr;
    std::string description;

    explicit implementation(int device) {
        const int count = ggml_backend_vk_get_device_count();
        if (device < 0 || device >= count) throw std::runtime_error("invalid Vulkan device index");
        char text[256] = {};
        ggml_backend_vk_get_device_description(device, text, sizeof(text));
        description = text;
        backend = ggml_backend_vk_init(device);
        if (!backend) throw std::runtime_error("failed to initialize Vulkan backend");
    }
    ~implementation() {
        if (backend) ggml_backend_free(backend);
    }
};

pipeline::pipeline(int vulkan_device)
    : impl_(std::make_unique<implementation>(vulkan_device)) {}
pipeline::~pipeline() = default;

const std::string & pipeline::device_description() const { return impl_->description; }

generate_result pipeline::generate(const generate_options & options) {
    if (options.input_image.empty()) throw std::invalid_argument("input image is empty");
    if (options.steps < 1) throw std::invalid_argument("steps must be positive");
    if (options.num_gaussians < 32 || options.num_gaussians % 32 != 0) {
        throw std::invalid_argument("num-gaussians must be a positive multiple of 32");
    }
    const model_paths models = resolve_models(options.models);
    temporary_directory work(options.seed);
    work.keep = options.keep_temporary;
    deterministic_rng rng(options.seed);
    const auto started = std::chrono::steady_clock::now();

    rgba_image image = resize_short_side(load_rgba_image(options.input_image), kCanvas);
    image = remove_background(impl_->backend, std::move(image), models, work.path);
    rgba_image prepared = prepare_foreground(std::move(image), kCanvas, options.erode_radius);
    const std::vector<float> raw_pixels = image_chw(prepared, false);
    std::vector<float> normalized(raw_pixels.size());
    const size_t plane = size_t(kCanvas) * kCanvas;
    for (int c = 0; c < 3; ++c) for (size_t i = 0; i < plane; ++i) {
        normalized[size_t(c) * plane + i] =
            (raw_pixels[size_t(c) * plane + i] - kMean[c]) / kStd[c];
    }

    std::vector<float> rope_cos, rope_sin;
    make_dino_rope(rope_cos, rope_sin);
    const fs::path dino_input = work.path / "dino-input.safetensors";
    const fs::path dino_output = work.path / "dino-output.safetensors";
    save_safetensors(dino_input.string(), {
        f32_tensor("pixels", {1, 3, kCanvas, kCanvas}, normalized),
        f32_tensor("rope_cos", {1, 1, 4096, 64}, rope_cos),
        f32_tensor("rope_sin", {1, 1, 4096, 64}, rope_sin),
    });
    invoke_worker(impl_->backend, {"--run-dino", models.dino,
                                  dino_input.string(), dino_output.string()});
    std::vector<float> feature1 =
        as_f32(load_host_safetensors(dino_output.string()).at("feature1"));
    if (feature1.size() % 1280 != 0) throw std::runtime_error("invalid DINO feature shape");
    for (size_t token = 0; token < feature1.size() / 1280; ++token) {
        float * row = feature1.data() + token * 1280;
        double mean = 0, variance = 0;
        for (int i = 0; i < 1280; ++i) mean += row[i];
        mean /= 1280;
        for (int i = 0; i < 1280; ++i) variance += (row[i] - mean) * (row[i] - mean);
        variance /= 1280;
        const float scale = 1.0f / std::sqrt(float(variance) + 1e-5f);
        for (int i = 0; i < 1280; ++i) row[i] = (row[i] - float(mean)) * scale;
    }

    std::vector<float> vae_image(raw_pixels.size()), noise(size_t(32) * 128 * 128);
    for (size_t i = 0; i < raw_pixels.size(); ++i) vae_image[i] = raw_pixels[i] * 2 - 1;
    rng.fill_normal(noise.data(), noise.size());
    const fs::path vae_input = work.path / "vae-input.safetensors";
    const fs::path vae_output = work.path / "vae-output.safetensors";
    save_safetensors(vae_input.string(), {
        f32_tensor("image", {1, 3, kCanvas, kCanvas}, vae_image),
        f32_tensor("noise", {1, 32, 128, 128}, noise),
    });
    invoke_worker(impl_->backend, {"--run-vae", models.vae_encoder,
                                  vae_input.string(), vae_output.string()});
    std::vector<float> encoded =
        as_f32(load_host_safetensors(vae_output.string()).at("feature2"));
    if (encoded.size() % 128 != 0) throw std::runtime_error("invalid VAE feature shape");
    std::vector<float> feature2(size_t(5) * 128, 0.0f);
    feature2.insert(feature2.end(), encoded.begin(), encoded.end());
    const size_t tokens = feature1.size() / 1280;
    if (feature2.size() / 128 != tokens) throw std::runtime_error("encoder token mismatch");

    std::vector<float> latent(size_t(8192) * 16), camera(5);
    rng.fill_normal(latent.data(), latent.size());
    rng.fill_normal(camera.data(), camera.size());
    const std::string asset_dir =
        options.asset_directory.empty() ? "assets" : options.asset_directory;
    const host_tensor_archive position_archive = load_host_safetensors(
        (fs::path(asset_dir) / "flow_positions.safetensors").string());
    const host_tensor & positions = position_archive.at("positions");
    if (positions.bytes.size() != positions.expected_byte_count()) {
        throw std::runtime_error("invalid bundled Flow positions payload");
    }
    const fs::path flow_input = work.path / "flow-input.safetensors";
    const fs::path flow_output = work.path / "flow-output.safetensors";
    save_safetensors(flow_input.string(), {
        f32_tensor("latent", {1, 8192, 16}, latent),
        f32_tensor("camera", {1, 1, 5}, camera),
        f32_tensor("feature1", {1, int64_t(tokens), 1280}, feature1),
        f32_tensor("feature2", {1, int64_t(tokens), 128}, feature2),
        f32_tensor("timestep", {1}, std::vector<float>(1)),
        copy_host_tensor("positions", positions.type, positions.shape,
                         positions.bytes.data(), positions.bytes.size()),
        f32_tensor("position_freqs", {170}, position_freqs_v1()),
    });
    invoke_worker(impl_->backend, {"--run-flow", models.flow, flow_input.string(),
                                  flow_output.string(), std::to_string(options.steps),
                                  std::to_string(options.guidance)});
    latent = as_f32(load_host_safetensors(flow_output.string()).at("latent"));

    const fs::path decode_input = work.path / "decode-input.safetensors";
    save_safetensors(decode_input.string(), {
        f32_tensor("condition", {1, 8192, 16}, latent),
        f32_tensor("position_freqs", {170}, position_freqs_v2()),
    });
    fs::path prefix = fs::absolute(options.output_prefix);
    if (!prefix.parent_path().empty()) fs::create_directories(prefix.parent_path());
    const uint64_t decode_seed = (uint64_t(rng.next_u32()) << 32) | rng.next_u32();
    invoke_worker(impl_->backend, {"--run-decode", models.decoder, decode_input.string(),
                                  prefix.string(), std::to_string(options.num_gaussians),
                                  std::to_string(decode_seed)});

    generate_result result;
    result.ply_path = prefix.string() + ".ply";
    result.splat_path = prefix.string() + ".splat";
    result.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (work.keep) std::printf("kept temporary tensors in %s\n", work.path.c_str());
    return result;
}

} // namespace triposplat
