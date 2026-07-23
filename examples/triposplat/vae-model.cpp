#include "vae-model.h"
#include "triposplat-ops.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace triposplat {
namespace {

bool persistent_f16_enabled() {
    const char * value = std::getenv("TRIPOSPLAT_VAE_PERSISTENT_F16");
    return value == nullptr || std::string(value) != "0";
}

ggml_tensor * channel_parameter(ggml_context * ctx, ggml_tensor * value) {
    return ggml_reshape_4d(ctx, value, 1, 1, value->ne[0], 1);
}

ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * x, const weight_store & weights,
                     const std::string & prefix, int stride = 1, int padding = 0) {
    ggml_tensor * y = persistent_f16_enabled()
        ? ggml_conv_2d_direct(ctx, weights.get(prefix + ".weight"), x,
                              stride, stride, padding, padding, 1, 1)
        : ggml_conv_2d(ctx, weights.get(prefix + ".weight"), x,
                       stride, stride, padding, padding, 1, 1);
    return ggml_add(ctx, y, channel_parameter(ctx, weights.get(prefix + ".bias")));
}

ggml_tensor * group_norm(ggml_context * ctx, ggml_tensor * x, const weight_store & weights,
                         const std::string & prefix) {
    ggml_tensor * y = ggml_group_norm(ctx, x, 32, 1e-6f);
    y = ggml_mul(ctx, y, channel_parameter(ctx, weights.get(prefix + ".weight")));
    return ggml_add(ctx, y, channel_parameter(ctx, weights.get(prefix + ".bias")));
}

ggml_tensor * resnet(ggml_context * ctx, ggml_tensor * x, const weight_store & weights,
                     const std::string & prefix, bool shortcut) {
    ggml_tensor * h = ggml_silu(ctx, group_norm(ctx, x, weights, prefix + ".norm1"));
    h = conv2d(ctx, h, weights, prefix + ".conv1", 1, 1);
    h = ggml_silu(ctx, group_norm(ctx, h, weights, prefix + ".norm2"));
    h = conv2d(ctx, h, weights, prefix + ".conv2", 1, 1);
    if (shortcut) x = conv2d(ctx, x, weights, prefix + ".conv_shortcut");
    return ggml_add(ctx, h, x);
}

ggml_tensor * downsample(ggml_context * ctx, ggml_tensor * x, const weight_store & weights,
                         const std::string & prefix) {
    if (persistent_f16_enabled()) x = ggml_cast(ctx, x, GGML_TYPE_F32);
    x = ggml_pad(ctx, x, 1, 1, 0, 0);
    if (persistent_f16_enabled()) x = ggml_cast(ctx, x, GGML_TYPE_F16);
    return conv2d(ctx, x, weights, prefix + ".conv", 2, 0);
}

ggml_tensor * mid_attention(ggml_context * ctx, ggml_tensor * x,
                            const weight_store & weights, const std::string & prefix) {
    const int64_t width = x->ne[0], height = x->ne[1], channels = x->ne[2], batch = x->ne[3];
    ggml_tensor * h = group_norm(ctx, x, weights, prefix + ".group_norm");
    h = ggml_permute(ctx, h, 1, 2, 0, 3); // [W,H,C,B] -> [C,W,H,B]
    h = ggml_cont(ctx, h);
    h = ggml_reshape_3d(ctx, h, channels, width * height, batch);
    auto projection = [&](const char * name) {
        ggml_tensor * p = linear(ctx, h, weights, prefix + "." + name);
        return ggml_reshape_4d(ctx, p, channels, width * height, 1, batch);
    };
    ggml_tensor * q = projection("to_q");
    ggml_tensor * k = projection("to_k");
    ggml_tensor * v = projection("to_v");
    k = attention_kv_storage(ctx, k);
    v = attention_kv_storage(ctx, v);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                             1.0f / std::sqrt(float(channels)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_3d(ctx, out, channels, width * height, batch);
    out = linear(ctx, out, weights, prefix + ".to_out.0");
    out = ggml_reshape_4d(ctx, out, channels, width, height, batch);
    out = ggml_permute(ctx, out, 2, 0, 1, 3); // [C,W,H,B] -> [W,H,C,B]
    out = ggml_cont(ctx, out);
    if (persistent_f16_enabled()) out = ggml_cast(ctx, out, GGML_TYPE_F16);
    return ggml_add(ctx, x, out);
}

ggml_tensor * pixel_unshuffle_2(ggml_context * ctx, ggml_tensor * x) {
    const int64_t width = x->ne[0], height = x->ne[1], channels = x->ne[2], batch = x->ne[3];
    // Split the contiguous width axis into [x_parity, width/2]. A plain view
    // cannot express a stride on ne[0], so introducing this size-2 axis is
    // essential for selecting x=0,2,... and x=1,3,... correctly.
    ggml_tensor * pairs = ggml_reshape_4d(ctx, x, 2, width/2, height, channels*batch);
    ggml_tensor * stacked = nullptr;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            ggml_tensor * part = ggml_view_4d(ctx, pairs, 1, width/2, height/2, channels*batch,
                                              pairs->nb[1], pairs->nb[2]*2, pairs->nb[3],
                                              size_t(dx)*pairs->nb[0] + size_t(dy)*pairs->nb[2]);
            part = ggml_cont(ctx, part);
            part = ggml_reshape_4d(ctx, part, width/2, height/2, channels, batch);
            part = ggml_permute(ctx, part, 1, 2, 0, 3); // [C,W,H,B]
            part = ggml_cont(ctx, part);
            part = ggml_reshape_4d(ctx, part, channels, 1, width/2, (height/2)*batch);
            stacked = stacked ? ggml_concat(ctx, stacked, part, 1) : part;
        }
    }
    stacked = ggml_permute(ctx, stacked, 1, 0, 2, 3); // [4,C,W,H*B]
    stacked = ggml_cont(ctx, stacked);
    return ggml_reshape_4d(ctx, stacked, channels*4, width/2, height/2, batch);
}

ggml_tensor * normalize_and_pack(ggml_context * ctx, const weight_store & weights,
                                 ggml_tensor * latent) {
    ggml_tensor * x = pixel_unshuffle_2(ctx, latent); // [128,W/2,H/2,B]
    ggml_tensor * mean = ggml_reshape_4d(ctx, weights.get("bn.running_mean"), 128, 1, 1, 1);
    ggml_tensor * variance = ggml_reshape_4d(ctx, weights.get("bn.running_var"), 128, 1, 1, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx, eps, 1e-5f);
    ggml_tensor * stddev = ggml_sqrt(ctx, ggml_add(ctx, variance, eps));
    x = ggml_div(ctx, ggml_sub(ctx, x, mean), stddev);
    return ggml_reshape_3d(ctx, x, 128, x->ne[1]*x->ne[2], x->ne[3]);
}

} // namespace

ggml_tensor * build_vae_encoder(ggml_context * ctx, const weight_store & weights,
                                ggml_tensor * image, int stop_after) {
    if (persistent_f16_enabled()) image = ggml_cast(ctx, image, GGML_TYPE_F16);
    ggml_tensor * x = conv2d(ctx, image, weights, "encoder.conv_in", 1, 1);
    if (stop_after == 0) return x;
    for (int i = 0; i < 2; ++i) x = resnet(ctx, x, weights, "encoder.down_blocks.0.resnets." + std::to_string(i), false);
    if (stop_after == 1) return x;
    x = downsample(ctx, x, weights, "encoder.down_blocks.0.downsamplers.0");
    if (stop_after == 2) return x;
    x = resnet(ctx, x, weights, "encoder.down_blocks.1.resnets.0", true);
    x = resnet(ctx, x, weights, "encoder.down_blocks.1.resnets.1", false);
    if (stop_after == 3) return x;
    x = downsample(ctx, x, weights, "encoder.down_blocks.1.downsamplers.0");
    if (stop_after == 4) return x;
    x = resnet(ctx, x, weights, "encoder.down_blocks.2.resnets.0", true);
    x = resnet(ctx, x, weights, "encoder.down_blocks.2.resnets.1", false);
    if (stop_after == 5) return x;
    x = downsample(ctx, x, weights, "encoder.down_blocks.2.downsamplers.0");
    if (stop_after == 6) return x;
    x = resnet(ctx, x, weights, "encoder.down_blocks.3.resnets.0", false);
    x = resnet(ctx, x, weights, "encoder.down_blocks.3.resnets.1", false);
    if (stop_after == 7) return x;
    x = resnet(ctx, x, weights, "encoder.mid_block.resnets.0", false);
    if (stop_after == 8) return x;
    x = mid_attention(ctx, x, weights, "encoder.mid_block.attentions.0");
    if (stop_after == 9) return x;
    x = resnet(ctx, x, weights, "encoder.mid_block.resnets.1", false);
    if (stop_after == 10) return x;
    x = ggml_silu(ctx, group_norm(ctx, x, weights, "encoder.conv_norm_out"));
    x = conv2d(ctx, x, weights, "encoder.conv_out", 1, 1);
    if (stop_after == 11) return x;
    x = conv2d(ctx, x, weights, "quant_conv");
    if (stop_after == 12) return x;

    // Deterministic parity uses the mean half of moments. Stochastic production
    // adds externally supplied noise before this packing step.
    x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], 32, x->ne[3],
                     x->nb[1], x->nb[2], x->nb[3], 0);
    return normalize_and_pack(ctx, weights, x);
}

ggml_tensor * build_vae_encoder_stochastic(ggml_context * ctx,
                                           const weight_store & weights,
                                           ggml_tensor * image,
                                           ggml_tensor * noise) {
    ggml_tensor * moments = build_vae_encoder(ctx, weights, image, 12);
    if (moments->ne[2] != 64 || noise->ne[0] != moments->ne[0] ||
        noise->ne[1] != moments->ne[1] || noise->ne[2] != 32 ||
        noise->ne[3] != moments->ne[3]) {
        throw std::runtime_error("VAE stochastic noise shape mismatch");
    }
    auto half = [&](int index) {
        ggml_tensor * value = ggml_view_4d(
            ctx, moments, moments->ne[0], moments->ne[1], 32, moments->ne[3],
            moments->nb[1], moments->nb[2], moments->nb[3],
            size_t(index) * 32 * moments->nb[2]);
        return ggml_cont(ctx, value);
    };
    ggml_tensor * mean = ggml_cast(ctx, half(0), GGML_TYPE_F32);
    ggml_tensor * logvar = ggml_cast(ctx, half(1), GGML_TYPE_F32);
    if (noise->type != GGML_TYPE_F32) noise = ggml_cast(ctx, noise, GGML_TYPE_F32);
    ggml_tensor * latent = ggml_add(
        ctx, mean, ggml_mul(ctx, ggml_exp(ctx, ggml_scale(ctx, logvar, 0.5f)), noise));
    if (latent->type != moments->type) latent = ggml_cast(ctx, latent, moments->type);
    return normalize_and_pack(ctx, weights, latent);
}

} // namespace triposplat
