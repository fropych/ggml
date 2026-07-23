#pragma once

#include "ggml.h"
#include "safetensors.h"

namespace triposplat {

struct gs_inputs {
    ggml_tensor * points = nullptr; // [3,L,B]
    ggml_tensor * condition = nullptr; // [16,Lc,B]
    ggml_tensor * position_freqs = nullptr; // [170]
};

ggml_tensor * build_gs_decoder(ggml_context * ctx, const weight_store & weights,
                               const gs_inputs & in, int blocks = 16);

struct octree_inputs {
    ggml_tensor * points = nullptr; // [3,L,B]
    ggml_tensor * level = nullptr; // [B], resolution value (2,4,...,256)
    ggml_tensor * condition = nullptr; // [16,Lc,B]
    ggml_tensor * position_freqs = nullptr;
};

ggml_tensor * build_octree_decoder(ggml_context * ctx, const weight_store & weights,
                                   const octree_inputs & in, int blocks = 4);

} // namespace triposplat
