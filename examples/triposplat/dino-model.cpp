#include "dino-model.h"
#include "triposplat-ops.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace triposplat {
namespace {

constexpr int kChannels = 1280;
constexpr int kHeads = 20;
constexpr int kHeadDim = 64;
constexpr int kPrefix = 5;

ggml_tensor * norm_f32(ggml_context * ctx, ggml_tensor * x,
                       const weight_store & weights, const std::string & prefix, float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    y = ggml_mul(ctx, y, ggml_cast(ctx, weights.get(prefix + ".weight"), GGML_TYPE_F32));
    return ggml_add(ctx, y, ggml_cast(ctx, weights.get(prefix + ".bias"), GGML_TYPE_F32));
}

ggml_tensor * attention(ggml_context * ctx, ggml_tensor * x,
                        ggml_tensor * cosine, ggml_tensor * sine,
                        const weight_store & weights, const std::string & prefix) {
    const int64_t tokens = x->ne[1];
    const int64_t batch = x->ne[2];
    auto project = [&](const char * name) {
        ggml_tensor * value = linear(ctx, x, weights, prefix + "." + name);
        return ggml_reshape_4d(ctx, value, kHeadDim, kHeads, tokens, batch);
    };
    const ggml_type kv_type = attention_kv_type();
    ggml_tensor * q = dino_rotary_fused(
        ctx, project("q_proj"), cosine, sine, kPrefix, GGML_TYPE_F32);
    ggml_tensor * k = dino_rotary_fused(
        ctx, project("k_proj"), cosine, sine, kPrefix,
        kv_type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);
    ggml_tensor * v = project("v_proj");
    k = attention_kv_storage(ctx, k);
    v = attention_kv_storage(ctx, v);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                             1.0f / std::sqrt(float(kHeadDim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_3d(ctx, out, kChannels, tokens, batch);
    return linear(ctx, out, weights, prefix + ".o_proj");
}

} // namespace

ggml_tensor * build_dino_model(ggml_context * ctx, const weight_store & weights,
                               const dino_inputs & in, int layers) {
    ggml_tensor * pixels_f32 = ggml_cast(ctx, in.pixels, GGML_TYPE_F32);
    ggml_tensor * patch = ggml_conv_2d(ctx, weights.get("embeddings.patch_embeddings.weight"),
                                       pixels_f32, 16, 16, 0, 0, 1, 1);
    ggml_tensor * patch_bias = ggml_reshape_4d(
        ctx, weights.get("embeddings.patch_embeddings.bias"), 1, 1, kChannels, 1);
    patch = ggml_add(ctx, patch, ggml_cast(ctx, patch_bias, GGML_TYPE_F32));
    const int64_t patch_count = patch->ne[0] * patch->ne[1];
    const int64_t batch = patch->ne[3];
    // ggml_permute arguments map each old axis to its new position.
    // [W,H,C,B] -> [C,W,H,B] is therefore (1,2,0,3).
    patch = ggml_permute(ctx, patch, 1, 2, 0, 3);
    patch = ggml_cont(ctx, patch);
    patch = ggml_reshape_3d(ctx, patch, kChannels, patch_count, batch);

    ggml_tensor * cls = ggml_repeat_4d(ctx, ggml_cast(ctx, weights.get("embeddings.cls_token"), GGML_TYPE_F32),
                                       kChannels, 1, batch, 1);
    ggml_tensor * reg = ggml_repeat_4d(ctx, ggml_cast(ctx, weights.get("embeddings.register_tokens"), GGML_TYPE_F32),
                                       kChannels, 4, batch, 1);
    ggml_tensor * x = ggml_concat(ctx, ggml_concat(ctx, cls, reg, 1), patch, 1);
    if (layers < 0) return x;
    for (int i = 0; i < layers; ++i) {
        const std::string p = "layer." + std::to_string(i);
        ggml_tensor * h = norm_f32(ctx, x, weights, p + ".norm1", 1e-5f);
        h = attention(ctx, h, in.rope_cos, in.rope_sin, weights, p + ".attention");
        h = ggml_mul(ctx, h, ggml_cast(ctx, weights.get(p + ".layer_scale1.lambda1"), GGML_TYPE_F32));
        x = ggml_add(ctx, x, h);
        h = norm_f32(ctx, x, weights, p + ".norm2", 1e-5f);
        ggml_tensor * gate = ggml_silu(ctx, linear(ctx, h, weights, p + ".mlp.gate_proj"));
        ggml_tensor * up = linear(ctx, h, weights, p + ".mlp.up_proj");
        h = linear(ctx, ggml_mul(ctx, gate, up), weights, p + ".mlp.down_proj");
        h = ggml_mul(ctx, h, ggml_cast(ctx, weights.get(p + ".layer_scale2.lambda1"), GGML_TYPE_F32));
        x = ggml_add(ctx, x, h);
    }
    return norm_f32(ctx, x, weights, "norm", 1e-5f);
}

} // namespace triposplat
