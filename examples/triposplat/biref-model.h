#pragma once

#include "ggml.h"
#include "safetensors.h"

#include <array>

namespace triposplat {

struct swin_inputs {
    ggml_tensor * pixels;
    // Shifted-window masks in Torch shape [num_windows, 144, 144].
    std::array<ggml_tensor *, 4> masks;
};

struct biref_inputs {
    ggml_tensor * pixels;
    std::array<ggml_tensor *, 4> full_masks;
    std::array<ggml_tensor *, 4> half_masks;
};

// Builds the Swin-L backbone through stop_stage and returns that stage's
// normalized NCHW feature in ggml order [W,H,C,N]. blocks_in_stop_stage can
// truncate the selected stage for layer-by-layer parity tests.
ggml_tensor * build_swin_backbone(ggml_context * ctx,
                                  const weight_store & weights,
                                  const swin_inputs & in,
                                  int stop_stage = 3,
                                  int blocks_in_stop_stage = -1);

ggml_tensor * build_birefnet(ggml_context * ctx,
                             const weight_store & weights,
                             const biref_inputs & in);

} // namespace triposplat
