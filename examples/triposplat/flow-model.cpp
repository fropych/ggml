#include "flow-model.h"
#include "triposplat-ops.h"

#include <cmath>
#include <string>

namespace triposplat {
namespace {

constexpr int kChannels = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;

ggml_tensor * position_embedding(ggml_context * ctx, const flow_inputs & in) {
    const int64_t length = in.positions->ne[1];
    const int64_t batch = in.positions->ne[2];
    ggml_tensor * p = ggml_reshape_4d(ctx, in.positions, 1, 3, length, batch);
    p = ggml_repeat_4d(ctx, p, 170, 3, length, batch);
    ggml_tensor * f = ggml_repeat_4d(ctx, in.position_freqs, 170, 3, length, batch);
    ggml_tensor * angle = ggml_scale(ctx, ggml_mul(ctx, p, f), 2.0f * float(M_PI));
    ggml_tensor * embed = ggml_concat(ctx, ggml_sin(ctx, angle), ggml_cos(ctx, angle), 0);
    embed = ggml_reshape_3d(ctx, embed, 1020, length, batch);
    ggml_tensor * tail = ggml_new_tensor_3d(ctx, embed->type, 4, length, batch);
    tail = ggml_fill(ctx, tail, 0.0f);
    return ggml_concat(ctx, embed, tail, 0);
}

ggml_tensor * modulation_part(ggml_context * ctx, ggml_tensor * mod, int index,
                              int64_t batch) {
    ggml_tensor * part = ggml_view_2d(ctx, mod, kChannels, batch, mod->nb[1],
                                      size_t(index) * size_t(kChannels) * ggml_element_size(mod));
    part = ggml_cont(ctx, part);
    // Elementwise Vulkan kernels broadcast this across the token dimension.
    return ggml_reshape_3d(ctx, part, kChannels, 1, batch);
}

ggml_tensor * modulated_block(ggml_context * ctx, ggml_tensor * x, ggml_tensor * shared_mod,
                              const weight_store & weights, const std::string & prefix,
                              const std::string & rope_prefix) {
    const int64_t batch = x->ne[2];
    ggml_tensor * mod = ggml_add(ctx, shared_mod, weights.get(prefix + ".shift_table"));
    ggml_tensor * shift_msa = modulation_part(ctx, mod, 0, batch);
    ggml_tensor * scale_msa = modulation_part(ctx, mod, 1, batch);
    ggml_tensor * gate_msa  = modulation_part(ctx, mod, 2, batch);
    ggml_tensor * shift_mlp = modulation_part(ctx, mod, 3, batch);
    ggml_tensor * scale_mlp = modulation_part(ctx, mod, 4, batch);
    ggml_tensor * gate_mlp  = modulation_part(ctx, mod, 5, batch);

    ggml_tensor * h = layer_norm_no_affine(ctx, x, 1e-6f);
    h = ggml_add(ctx, ggml_add(ctx, h, ggml_mul(ctx, h, scale_msa)), shift_msa);
    ggml_tensor * angles = repo_rotary(ctx, x, weights, rope_prefix, kHeads, kHeadDim);
    h = self_attention(ctx, h, angles, weights, prefix + ".attn", kHeads, true);
    x = ggml_add(ctx, x, ggml_mul(ctx, h, gate_msa));

    h = layer_norm_no_affine(ctx, x, 1e-6f);
    h = ggml_add(ctx, ggml_add(ctx, h, ggml_mul(ctx, h, scale_mlp)), shift_mlp);
    h = feed_forward_gelu(ctx, h, weights, prefix + ".mlp");
    return ggml_add(ctx, x, ggml_mul(ctx, h, gate_mlp));
}

ggml_tensor * plain_block(ggml_context * ctx, ggml_tensor * x,
                          const weight_store & weights, const std::string & prefix,
                          const std::string & rope_prefix) {
    ggml_tensor * angles = repo_rotary(ctx, x, weights, rope_prefix, kHeads, kHeadDim);
    ggml_tensor * h = layer_norm(ctx, x, weights, prefix + ".norm1", 1e-6f);
    x = ggml_add(ctx, x, self_attention(ctx, h, angles, weights, prefix + ".attn", kHeads, true));
    h = layer_norm(ctx, x, weights, prefix + ".norm2", 1e-6f);
    return ggml_add(ctx, x, feed_forward_gelu(ctx, h, weights, prefix + ".mlp"));
}

} // namespace

flow_outputs build_flow_outputs(ggml_context * ctx, const weight_store & weights,
                                const flow_inputs & in, int blocks) {
    const int64_t batch = in.latent->ne[2];
    ggml_tensor * hx = linear(ctx, in.latent, weights, "input_layer");
    hx = ggml_add(ctx, hx, position_embedding(ctx, in));
    ggml_tensor * hc = linear(ctx, in.feature1, weights, "cond_embedder");
    hc = ggml_add(ctx, hc, linear(ctx, in.feature2, weights, "cond_embedder2"));

    ggml_tensor * temb = ggml_timestep_embedding(ctx, in.timestep, 256, 10000);
    temb = linear(ctx, temb, weights, "t_embedder.mlp.0");
    temb = ggml_silu(ctx, temb);
    temb = linear(ctx, temb, weights, "t_embedder.mlp.2");
    ggml_tensor * tmod = linear(ctx, ggml_silu(ctx, temb), weights, "adaLN_modulation.1");

    for (int i = 0; i < 2; ++i) {
        hx = modulated_block(ctx, hx, tmod, weights,
                             "noise_refiner." + std::to_string(i),
                             "noise_repo_layers." + std::to_string(i));
        hc = plain_block(ctx, hc, weights,
                         "context_refiner." + std::to_string(i),
                         "context_repo_layers." + std::to_string(i));
    }

    ggml_tensor * hcam = linear(ctx, in.camera, weights, "cam_refiner.mlp.0");
    hcam = ggml_gelu(ctx, hcam);
    hcam = linear(ctx, hcam, weights, "cam_refiner.mlp.2");
    ggml_tensor * h = ggml_concat(ctx, ggml_concat(ctx, hx, hc, 1), hcam, 1);
    for (int i = 0; i < blocks; ++i) {
        h = modulated_block(ctx, h, tmod, weights,
                            "blocks." + std::to_string(i),
                            "repo_layers." + std::to_string(i));
    }

    hx = ggml_view_3d(ctx, h, kChannels, in.latent->ne[1], batch,
                      h->nb[1], h->nb[2], 0);
    hx = layer_norm_no_affine(ctx, hx, 1e-6f);
    ggml_tensor * final_mod = ggml_add(ctx, weights.get("shift_table"),
                                       ggml_reshape_3d(ctx, temb, kChannels, 1, batch));
    ggml_tensor * shift = ggml_view_3d(ctx, final_mod, kChannels, 1, batch,
                                       final_mod->nb[1], final_mod->nb[2], 0);
    ggml_tensor * scale = ggml_view_3d(ctx, final_mod, kChannels, 1, batch,
                                       final_mod->nb[1], final_mod->nb[2], final_mod->nb[1]);
    hx = ggml_add(ctx, ggml_add(ctx, hx, ggml_mul(ctx, hx, scale)), shift);
    ggml_tensor * latent_out=linear(ctx,hx,weights,"out_layer");
    hcam=ggml_view_3d(ctx,h,kChannels,in.camera->ne[1],batch,h->nb[1],h->nb[2],
        size_t(in.latent->ne[1]+in.feature1->ne[1])*h->nb[1]);
    hcam=layer_norm_no_affine(ctx,hcam,1e-6f);
    hcam=ggml_add(ctx,ggml_add(ctx,hcam,ggml_mul(ctx,hcam,scale)),shift);
    return {latent_out,linear(ctx,hcam,weights,"cam_out_layer")};
}

ggml_tensor * build_flow_model(ggml_context * ctx, const weight_store & weights,
                               const flow_inputs & in, int blocks) {
    return build_flow_outputs(ctx,weights,in,blocks).latent;
}

} // namespace triposplat
