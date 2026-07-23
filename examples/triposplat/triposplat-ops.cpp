#include "triposplat-ops.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace triposplat {

static constexpr int32_t deform_conv_2d_magic = 0x44434632; // "DCF2"
static constexpr int32_t image_to_patches_magic = 0x49545032; // "ITP2"
static constexpr int32_t dino_rotary_magic = 0x44524f54; // "DROT"

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x,
                     const weight_store & weights, const std::string & prefix) {
    ggml_tensor * w = weights.get(prefix + ".weight");
    if (w->ne[0] != x->ne[0]) {
        throw std::runtime_error(prefix + ": linear input dimension mismatch");
    }
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    if (ggml_tensor * b = weights.maybe(prefix + ".bias")) {
        y = ggml_add(ctx, y, b);
    }
    return y;
}

ggml_tensor * layer_norm_no_affine(ggml_context * ctx, ggml_tensor * x, float eps) {
    return ggml_norm(ctx, x, eps);
}

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                         const weight_store & weights, const std::string & prefix,
                         float eps) {
    ggml_tensor * y = ggml_norm(ctx, x, eps);
    if (ggml_tensor * w = weights.maybe(prefix + ".weight")) y = ggml_mul(ctx, y, w);
    if (ggml_tensor * b = weights.maybe(prefix + ".bias")) y = ggml_add(ctx, y, b);
    return y;
}

ggml_tensor * multihead_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                 const weight_store & weights, const std::string & prefix,
                                 float eps) {
    // PyTorch F.normalize(x, dim=-1) * sqrt(dim) is RMS normalization.
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, y, weights.get(prefix + ".gamma"));
}

ggml_tensor * feed_forward_gelu(ggml_context * ctx, ggml_tensor * x,
                                const weight_store & weights, const std::string & prefix) {
    x = linear(ctx, x, weights, prefix + ".mlp.0");
    x = ggml_gelu(ctx, x);
    return linear(ctx, x, weights, prefix + ".mlp.2");
}

ggml_tensor * repo_rotary(ggml_context * ctx, ggml_tensor * hidden,
                          const weight_store & weights, const std::string & prefix,
                          int heads, int head_dim) {
    ggml_tensor * h = layer_norm(ctx, hidden, weights, prefix + ".norm", 1e-6f);
    ggml_tensor * gate = ggml_silu(ctx, linear(ctx, h, weights, prefix + ".gate_map"));
    ggml_tensor * content = linear(ctx, h, weights, prefix + ".content_map");
    ggml_tensor * delta = linear(ctx, ggml_mul(ctx, gate, content), weights, prefix + ".final_map");
    const int64_t length = delta->ne[1];
    const int64_t batch = delta->ne[2];
    delta = ggml_reshape_4d(ctx, delta, 3, heads, length, batch);

    ggml_tensor * angles = nullptr;
    for (int axis = 0; axis < 3; ++axis) {
        const std::string frequency_name = prefix + ".freqs_" + std::to_string(axis);
        ggml_tensor * frequencies = weights.get(frequency_name);
        ggml_tensor * coordinate = ggml_view_4d(
            ctx, delta, 1, heads, length, batch,
            delta->nb[1], delta->nb[2], delta->nb[3], size_t(axis) * delta->nb[0]);
        coordinate = ggml_cont(ctx, coordinate);
        const int64_t nfreq = frequencies->ne[0];
        coordinate = ggml_repeat_4d(ctx, coordinate, nfreq, heads, length, batch);
        ggml_tensor * part = ggml_scale(ctx, ggml_mul(ctx, coordinate, frequencies), float(M_PI));
        angles = angles ? ggml_concat(ctx, angles, part, 0) : part;
    }
    if (angles->ne[0] * 2 != head_dim) {
        throw std::runtime_error(prefix + ": rotary frequency dimension mismatch");
    }
    return angles;
}

ggml_tensor * apply_rotary(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * angles) {
    const int64_t head_dim = x->ne[0];
    const int64_t heads = x->ne[1];
    const int64_t length = x->ne[2];
    const int64_t batch = x->ne[3];
    if (head_dim % 2 != 0 || angles->ne[0] != head_dim / 2) {
        throw std::runtime_error("invalid dynamic rotary dimensions");
    }
    ggml_tensor * pairs = ggml_reshape_4d(ctx, x, 2, head_dim / 2, heads, length * batch);
    auto component = [&](int index) {
        return ggml_view_4d(ctx, pairs, 1, head_dim / 2, heads, length * batch,
                            pairs->nb[1], pairs->nb[2], pairs->nb[3],
                            size_t(index) * pairs->nb[0]);
    };
    ggml_tensor * even = component(0);
    ggml_tensor * odd = component(1);
    ggml_tensor * angle = ggml_reshape_4d(ctx, angles, 1, head_dim / 2, heads, length * batch);
    ggml_tensor * cosine = ggml_cos(ctx, angle);
    ggml_tensor * sine = ggml_sin(ctx, angle);
    ggml_tensor * out_even = ggml_sub(ctx, ggml_mul(ctx, even, cosine), ggml_mul(ctx, odd, sine));
    ggml_tensor * out_odd = ggml_add(ctx, ggml_mul(ctx, even, sine), ggml_mul(ctx, odd, cosine));
    ggml_tensor * out = ggml_concat(ctx, out_even, out_odd, 0);
    return ggml_reshape_4d(ctx, out, head_dim, heads, length, batch);
}

ggml_type attention_kv_type() {
    const char * value = std::getenv("TRIPOSPLAT_ATTN_KV_TYPE");
    if (!value || std::strcmp(value, "f16") == 0) return GGML_TYPE_F16;
    if (std::strcmp(value, "f32") == 0) return GGML_TYPE_F32;
    if (std::strcmp(value, "bf16") == 0) return GGML_TYPE_BF16;
    throw std::runtime_error("TRIPOSPLAT_ATTN_KV_TYPE must be f16, f32, or bf16");
}

ggml_tensor * attention_kv_storage(ggml_context * ctx, ggml_tensor * x) {
    const ggml_type target = attention_kv_type();
    return x->type == target ? x : ggml_cast(ctx, x, target);
}

ggml_tensor * self_attention(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * angles, const weight_store & weights,
                             const std::string & prefix, int heads, bool qk_rms_norm) {
    const int64_t channels = x->ne[0];
    const int64_t length = x->ne[1];
    const int64_t batch = x->ne[2];
    const int64_t head_dim = channels / heads;
    ggml_tensor * qkv = linear(ctx, x, weights, prefix + ".qkv");
    const size_t element = ggml_element_size(qkv);
    auto projection = [&](int index) {
        ggml_tensor * view = ggml_view_3d(ctx, qkv, channels, length, batch,
                                          qkv->nb[1], qkv->nb[2],
                                          size_t(index) * size_t(channels) * element);
        view = ggml_cont(ctx, view);
        return ggml_reshape_4d(ctx, view, head_dim, heads, length, batch);
    };
    ggml_tensor * q = projection(0);
    ggml_tensor * k = projection(1);
    ggml_tensor * v = projection(2);
    if (angles) {
        q = apply_rotary(ctx, q, angles);
        k = apply_rotary(ctx, k, angles);
    }
    if (qk_rms_norm) {
        q = multihead_rms_norm(ctx, q, weights, prefix + ".q_norm");
        k = multihead_rms_norm(ctx, k, weights, prefix + ".k_norm");
    }
    k = attention_kv_storage(ctx, k);
    v = attention_kv_storage(ctx, v);
    // ggml flash attention consumes [D, tokens, heads, batch], whereas the
    // projection/rotary path above mirrors PyTorch's [batch,tokens,heads,D]
    // as contiguous ggml [D,heads,tokens,batch].
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                             1.0f / std::sqrt(float(head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_3d(ctx, out, channels, length, batch);
    return linear(ctx, out, weights, prefix + ".out");
}

ggml_tensor * deform_conv_2d(ggml_context * ctx,
                             ggml_tensor * input,
                             ggml_tensor * offsets,
                             ggml_tensor * masks,
                             ggml_tensor * weight,
                             int stride_w, int stride_h,
                             int pad_w, int pad_h) {
    GGML_ASSERT(input->type == GGML_TYPE_F32);
    GGML_ASSERT(offsets->type == GGML_TYPE_F32);
    GGML_ASSERT(masks->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16);
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(offsets));
    GGML_ASSERT(ggml_is_contiguous(masks));
    GGML_ASSERT(ggml_is_contiguous(weight));

    const int64_t kw = weight->ne[0];
    const int64_t kh = weight->ne[1];
    const int64_t channels = weight->ne[2];
    const int64_t out_channels = weight->ne[3];
    GGML_ASSERT(input->ne[2] == channels);
    GGML_ASSERT(offsets->ne[0] == masks->ne[0]);
    GGML_ASSERT(offsets->ne[1] == masks->ne[1]);
    GGML_ASSERT(offsets->ne[2] == 2 * kw * kh);
    GGML_ASSERT(masks->ne[2] == kw * kh);
    GGML_ASSERT(offsets->ne[3] == input->ne[3]);
    GGML_ASSERT(masks->ne[3] == input->ne[3]);

    ggml_tensor * samples = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, kw * kh * channels,
        offsets->ne[0] * offsets->ne[1], input->ne[3]);
    const int32_t params[] = {
        deform_conv_2d_magic, stride_w, stride_h, pad_w, pad_h,
        (int32_t) kw, (int32_t) kh,
    };
    static_assert(sizeof(params) <= GGML_MAX_OP_PARAMS, "deform conv params overflow");
    std::memcpy(samples->op_params, params, sizeof(params));
    samples->op = GGML_OP_CUSTOM;
    samples->src[0] = input;
    samples->src[1] = offsets;
    samples->src[2] = masks;
    ggml_tensor * matrix = ggml_reshape_2d(ctx, weight, kw * kh * channels, out_channels);
    ggml_tensor * result = ggml_mul_mat(ctx, matrix, samples); // [OC,OW*OH,N]
    result = ggml_reshape_4d(ctx, result, out_channels,
                             offsets->ne[0], offsets->ne[1], input->ne[3]);
    result = ggml_permute(ctx, result, 2, 0, 1, 3); // [OC,W,H,N] -> [W,H,OC,N]
    return ggml_cont(ctx, result);
}

ggml_tensor * image_to_patches(ggml_context * ctx, ggml_tensor * image,
                               int64_t patch_width, int64_t patch_height) {
    GGML_ASSERT(image->type == GGML_TYPE_F32 && ggml_is_contiguous(image));
    GGML_ASSERT(image->ne[0] % patch_width == 0 && image->ne[1] % patch_height == 0);
    const int64_t wg=image->ne[0]/patch_width, hg=image->ne[1]/patch_height;
    ggml_tensor * result=ggml_new_tensor_4d(
        ctx,GGML_TYPE_F32,patch_width,patch_height,image->ne[2]*wg*hg,image->ne[3]);
    const int32_t params[]={image_to_patches_magic,(int32_t)wg,(int32_t)hg};
    std::memcpy(result->op_params,params,sizeof(params));
    result->op=GGML_OP_CUSTOM;result->src[0]=image;
    return result;
}

ggml_tensor * dino_rotary_fused(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * cosine, ggml_tensor * sine,
                                int prefix_tokens, ggml_type output_type) {
    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT((cosine->type == GGML_TYPE_F32 || cosine->type == GGML_TYPE_F16) &&
                sine->type == cosine->type);
    GGML_ASSERT(ggml_is_contiguous(cosine) && ggml_is_contiguous(sine));
    GGML_ASSERT(output_type == GGML_TYPE_F32 || output_type == GGML_TYPE_F16);
    GGML_ASSERT(x->ne[0] % 2 == 0 && prefix_tokens >= 0 && prefix_tokens < x->ne[2]);
    GGML_ASSERT(cosine->ne[0] == x->ne[0] && sine->ne[0] == x->ne[0]);
    GGML_ASSERT(cosine->ne[1] == x->ne[2] - prefix_tokens &&
                sine->ne[1] == x->ne[2] - prefix_tokens);
    GGML_ASSERT(cosine->ne[2] == 1 && cosine->ne[3] == 1 &&
                ggml_are_same_shape(cosine, sine));

    ggml_tensor * result = ggml_new_tensor_4d(
        ctx, output_type, x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    const int32_t params[] = { dino_rotary_magic, prefix_tokens };
    std::memcpy(result->op_params, params, sizeof(params));
    result->op = GGML_OP_CUSTOM;
    result->src[0] = x;
    result->src[1] = cosine;
    result->src[2] = sine;
    return result;
}

} // namespace triposplat
