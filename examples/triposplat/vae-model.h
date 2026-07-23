#pragma once

#include "ggml.h"
#include "safetensors.h"

namespace triposplat {

// Input uses ggml image layout [W,H,3,B]. Output uses [128,tokens,B].
ggml_tensor * build_vae_encoder(ggml_context * ctx, const weight_store & weights,
                                ggml_tensor * image, int stop_after = 13);

// Production encoder path. `noise` uses image layout [W,H,32,B] at the
// post-downsample resolution and is consumed as mean + exp(0.5*logvar)*noise.
// The returned feature tensor uses [128,tokens,B], matching feature2.
ggml_tensor * build_vae_encoder_stochastic(ggml_context * ctx,
                                           const weight_store & weights,
                                           ggml_tensor * image,
                                           ggml_tensor * noise);

} // namespace triposplat
