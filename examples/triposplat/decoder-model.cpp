#include "decoder-model.h"
#include "triposplat-ops.h"

#include <cmath>
#include <string>

namespace triposplat {
namespace {

constexpr int kChannels = 1024;
constexpr int kHeads = 16;
constexpr int kHeadDim = 64;

ggml_tensor * position_v2(ggml_context * ctx, const gs_inputs & in) {
    const int64_t length=in.points->ne[1], batch=in.points->ne[2];
    ggml_tensor * p=ggml_reshape_4d(ctx,in.points,1,3,length,batch);
    p=ggml_repeat_4d(ctx,p,170,3,length,batch);
    ggml_tensor * f=ggml_repeat_4d(ctx,in.position_freqs,170,3,length,batch);
    ggml_tensor * a=ggml_scale(ctx,ggml_mul(ctx,p,f),float(M_PI));
    ggml_tensor * e=ggml_concat(ctx,ggml_sin(ctx,a),ggml_cos(ctx,a),0);
    e=ggml_reshape_3d(ctx,e,1020,length,batch);
    ggml_tensor * tail=ggml_new_tensor_3d(ctx,e->type,4,length,batch);
    tail=ggml_fill(ctx,tail,0.0f);
    return ggml_concat(ctx,e,tail,0);
}

ggml_tensor * position_v2_raw(ggml_context * ctx, ggml_tensor * points, ggml_tensor * freqs) {
    gs_inputs in { points, nullptr, freqs };
    return position_v2(ctx, in);
}

ggml_tensor * self_attn(ggml_context * ctx, ggml_tensor * x, const weight_store & w,
                        const std::string & p) {
    const int64_t length=x->ne[1], batch=x->ne[2];
    ggml_tensor * qkv=linear(ctx,x,w,p+".to_qkv");
    const size_t el=ggml_element_size(qkv);
    auto get=[&](int i) {
        ggml_tensor * t=ggml_view_3d(ctx,qkv,kChannels,length,batch,qkv->nb[1],qkv->nb[2],
                                     size_t(i)*kChannels*el);
        t=ggml_cont(ctx,t);
        return ggml_reshape_4d(ctx,t,kHeadDim,kHeads,length,batch);
    };
    ggml_tensor * q=multihead_rms_norm(ctx,get(0),w,p+".q_rms_norm");
    ggml_tensor * k=multihead_rms_norm(ctx,get(1),w,p+".k_rms_norm");
    ggml_tensor * v=get(2);
    k=attention_kv_storage(ctx,k); v=attention_kv_storage(ctx,v);
    q=ggml_permute(ctx,q,0,2,1,3); k=ggml_permute(ctx,k,0,2,1,3); v=ggml_permute(ctx,v,0,2,1,3);
    ggml_tensor * y=ggml_flash_attn_ext(ctx,q,k,v,nullptr,1.0f/std::sqrt(float(kHeadDim)),0,0);
    ggml_flash_attn_ext_set_prec(y,GGML_PREC_F32);
    y=ggml_reshape_3d(ctx,y,kChannels,length,batch);
    return linear(ctx,y,w,p+".to_out");
}

ggml_tensor * cross_attn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * context,
                         const weight_store & w, const std::string & p) {
    const int64_t length=x->ne[1], kvlen=context->ne[1], batch=x->ne[2];
    ggml_tensor * q=linear(ctx,x,w,p+".to_q");
    q=ggml_reshape_4d(ctx,q,kHeadDim,kHeads,length,batch);
    q=multihead_rms_norm(ctx,q,w,p+".q_rms_norm");
    ggml_tensor * kv=linear(ctx,context,w,p+".to_kv");
    const size_t el=ggml_element_size(kv);
    auto get=[&](int i) {
        ggml_tensor * t=ggml_view_3d(ctx,kv,kChannels,kvlen,batch,kv->nb[1],kv->nb[2],
                                     size_t(i)*kChannels*el);
        t=ggml_cont(ctx,t);
        return ggml_reshape_4d(ctx,t,kHeadDim,kHeads,kvlen,batch);
    };
    ggml_tensor * k=multihead_rms_norm(ctx,get(0),w,p+".k_rms_norm");
    ggml_tensor * v=get(1);
    k=attention_kv_storage(ctx,k); v=attention_kv_storage(ctx,v);
    q=ggml_permute(ctx,q,0,2,1,3); k=ggml_permute(ctx,k,0,2,1,3); v=ggml_permute(ctx,v,0,2,1,3);
    ggml_tensor * y=ggml_flash_attn_ext(ctx,q,k,v,nullptr,1.0f/std::sqrt(float(kHeadDim)),0,0);
    ggml_flash_attn_ext_set_prec(y,GGML_PREC_F32);
    y=ggml_reshape_3d(ctx,y,kChannels,length,batch);
    return linear(ctx,y,w,p+".to_out");
}

} // namespace

ggml_tensor * build_gs_decoder(ggml_context * ctx, const weight_store & weights,
                               const gs_inputs & in, int blocks) {
    ggml_tensor * h=ggml_add(ctx,linear(ctx,in.points,weights,"gs.in_proj"),position_v2(ctx,in));
    h=linear(ctx,h,weights,"gs.input_layer");
    for (int i=0;i<blocks;++i) {
        const std::string p="gs.blocks."+std::to_string(i);
        ggml_tensor * x=layer_norm_no_affine(ctx,h,1e-6f);
        h=ggml_add(ctx,h,self_attn(ctx,x,weights,p+".self_attn"));
        x=layer_norm(ctx,h,weights,p+".norm2",1e-6f);
        h=ggml_add(ctx,h,cross_attn(ctx,x,in.condition,weights,p+".cross_attn"));
        x=layer_norm_no_affine(ctx,h,1e-6f);
        h=ggml_add(ctx,h,feed_forward_gelu(ctx,x,weights,p+".mlp"));
    }
    h=layer_norm_no_affine(ctx,h,1e-6f);
    return linear(ctx,h,weights,"gs.out_proj");
}

ggml_tensor * build_octree_decoder(ggml_context * ctx, const weight_store & weights,
                                   const octree_inputs & in, int blocks) {
    const int64_t length=in.points->ne[1], batch=in.points->ne[2];
    ggml_tensor * h=ggml_add(ctx,linear(ctx,in.points,weights,"octree.in_proj"),
                             position_v2_raw(ctx,in.points,in.position_freqs));
    h=linear(ctx,h,weights,"octree.input_layer");
    ggml_tensor * level=ggml_scale(ctx,in.level,2.0f*float(M_PI));
    ggml_tensor * mod=ggml_timestep_embedding(ctx,level,256,1024);
    mod=linear(ctx,mod,weights,"octree.l_embedder.mlp.0");
    mod=ggml_silu(ctx,mod);
    mod=linear(ctx,mod,weights,"octree.l_embedder.mlp.2");
    mod=linear(ctx,ggml_silu(ctx,mod),weights,"octree.adaLN_modulation.1");
    auto part=[&](int index) {
        ggml_tensor * p=ggml_view_2d(ctx,mod,kChannels,batch,mod->nb[1],
                                     size_t(index)*kChannels*ggml_element_size(mod));
        p=ggml_cont(ctx,p);
        return ggml_repeat_4d(ctx,p,kChannels,length,batch,1);
    };
    ggml_tensor * shift_a=part(0),*scale_a=part(1),*gate_a=part(2);
    ggml_tensor * shift_m=part(3),*scale_m=part(4),*gate_m=part(5);
    for(int i=0;i<blocks;++i){
        const std::string p="octree.blocks."+std::to_string(i);
        ggml_tensor * x=layer_norm_no_affine(ctx,h,1e-6f);
        x=ggml_add(ctx,ggml_add(ctx,x,ggml_mul(ctx,x,scale_a)),shift_a);
        x=cross_attn(ctx,x,in.condition,weights,p+".cross_attn");
        h=ggml_add(ctx,h,ggml_mul(ctx,x,gate_a));
        x=layer_norm_no_affine(ctx,h,1e-6f);
        x=ggml_add(ctx,ggml_add(ctx,x,ggml_mul(ctx,x,scale_m)),shift_m);
        x=feed_forward_gelu(ctx,x,weights,p+".mlp");
        h=ggml_add(ctx,h,ggml_mul(ctx,x,gate_m));
    }
    h=layer_norm_no_affine(ctx,h,1e-6f);
    return linear(ctx,h,weights,"octree.out_proj");
}

} // namespace triposplat
