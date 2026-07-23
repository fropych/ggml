#pragma once

#include "ggml.h"
#include "safetensors.h"

namespace triposplat {

struct dino_inputs {
    ggml_tensor * pixels = nullptr; // [W,H,3,B]
    ggml_tensor * rope_cos = nullptr; // [64,patches,1,1]
    ggml_tensor * rope_sin = nullptr;
};

ggml_tensor * build_dino_model(ggml_context * ctx, const weight_store & weights,
                               const dino_inputs & in, int layers = 32);

} // namespace triposplat
