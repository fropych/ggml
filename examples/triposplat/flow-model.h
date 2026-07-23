#pragma once

#include "ggml.h"
#include "safetensors.h"

namespace triposplat {

struct flow_inputs {
    // ggml layouts corresponding to PyTorch [B,L,C] are [C,L,B].
    ggml_tensor * latent = nullptr;      // [16, Lz, B]
    ggml_tensor * feature1 = nullptr;    // [1280, Lc, B]
    ggml_tensor * feature2 = nullptr;    // [128, Lc, B]
    ggml_tensor * camera = nullptr;      // [5, Lcam, B]
    ggml_tensor * timestep = nullptr;    // [B]
    ggml_tensor * positions = nullptr;   // [3, Lz, B]
    ggml_tensor * position_freqs = nullptr; // [170]
};

struct flow_outputs {
    ggml_tensor * latent = nullptr;
    ggml_tensor * camera = nullptr;
};

flow_outputs build_flow_outputs(ggml_context * ctx, const weight_store & weights,
                                const flow_inputs & in, int blocks = 24);

ggml_tensor * build_flow_model(ggml_context * ctx, const weight_store & weights,
                               const flow_inputs & in, int blocks = 24);

} // namespace triposplat
