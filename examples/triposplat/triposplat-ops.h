#pragma once

#include "ggml.h"
#include "safetensors.h"

#include <string>

namespace triposplat {

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x,
                     const weight_store & weights, const std::string & prefix);
ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                         const weight_store & weights, const std::string & prefix,
                         float eps);
ggml_tensor * layer_norm_no_affine(ggml_context * ctx, ggml_tensor * x, float eps);
ggml_tensor * multihead_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                 const weight_store & weights, const std::string & prefix,
                                 float eps = 1e-12f);
ggml_tensor * feed_forward_gelu(ggml_context * ctx, ggml_tensor * x,
                                const weight_store & weights, const std::string & prefix);
ggml_tensor * repo_rotary(ggml_context * ctx, ggml_tensor * hidden,
                          const weight_store & weights, const std::string & prefix,
                          int heads, int head_dim);
ggml_tensor * apply_rotary(ggml_context * ctx, ggml_tensor * x,
                           ggml_tensor * angles);
ggml_type attention_kv_type();
ggml_tensor * attention_kv_storage(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * self_attention(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * angles, const weight_store & weights,
                             const std::string & prefix, int heads, bool qk_rms_norm);

// Modulated deformable convolution used by BiRefNet. All tensors are contiguous
// F32 in ggml order: input [W,H,C,N], offsets [OW,OH,2*KW*KH,N],
// masks [OW,OH,KW*KH,N], and weight [KW,KH,C,OC]. Dynamic sampling is a
// Vulkan-only custom op followed by ggml's tiled Vulkan matrix multiply.
ggml_tensor * deform_conv_2d(ggml_context * ctx,
                             ggml_tensor * input,
                             ggml_tensor * offsets,
                             ggml_tensor * masks,
                             ggml_tensor * weight,
                             int stride_w, int stride_h,
                             int pad_w, int pad_h);

ggml_tensor * image_to_patches(ggml_context * ctx, ggml_tensor * image,
                               int64_t patch_width, int64_t patch_height);

ggml_tensor * dino_rotary_fused(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * cosine, ggml_tensor * sine,
                                int prefix_tokens, ggml_type output_type);

} // namespace triposplat
