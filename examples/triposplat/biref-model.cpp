#include "biref-model.h"
#include "triposplat-ops.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace triposplat {
namespace {

constexpr int kWindow = 12;
constexpr int kWindowTokens = kWindow * kWindow;
constexpr int kDepths[4] = {2, 2, 18, 2};
constexpr int kHeads[4] = {6, 12, 24, 48};
constexpr int kChannels[4] = {192, 384, 768, 1536};

ggml_tensor * channel_parameter(ggml_context * ctx, ggml_tensor * value) {
    return ggml_reshape_4d(ctx, value, 1, 1, value->ne[0], 1);
}

ggml_tensor * tokens_to_image(ggml_context * ctx, ggml_tensor * x,
                              int64_t width, int64_t height) {
    const int64_t channels = x->ne[0], batch = x->ne[2];
    x = ggml_reshape_4d(ctx, x, channels, width, height, batch);
    x = ggml_permute(ctx, x, 2, 0, 1, 3); // [C,W,H,B] -> [W,H,C,B]
    return ggml_cont(ctx, x);
}

ggml_tensor * image_to_tokens(ggml_context * ctx, ggml_tensor * x) {
    const int64_t width=x->ne[0], height=x->ne[1], channels=x->ne[2], batch=x->ne[3];
    x = ggml_permute(ctx, x, 1, 2, 0, 3); // [W,H,C,B] -> [C,W,H,B]
    x = ggml_cont(ctx, x);
    return ggml_reshape_3d(ctx, x, channels, width*height, batch);
}

ggml_tensor * relative_mask(ggml_context * ctx, const weight_store & weights,
                            const std::string & prefix, ggml_tensor * shift_mask,
                            int heads, int64_t windows) {
    ggml_tensor * index = ggml_reshape_1d(
        ctx, weights.get(prefix + ".relative_position_index"),
        kWindowTokens*kWindowTokens);
    ggml_tensor * bias = ggml_get_rows(
        ctx, weights.get(prefix + ".relative_position_bias_table"), index);
    bias = ggml_reshape_4d(ctx, bias, heads, kWindowTokens, kWindowTokens, 1);
    bias = ggml_permute(ctx, bias, 2, 0, 1, 3); // [H,K,Q] -> [K,Q,H,1]
    bias = ggml_cont(ctx, bias);
    bias = ggml_cast(ctx, bias, GGML_TYPE_F16);
    if (!shift_mask) return bias;
    ggml_tensor * mask = ggml_reshape_4d(
        ctx, ggml_cast(ctx, shift_mask, GGML_TYPE_F16),
        kWindowTokens, kWindowTokens, 1, windows);
    bias = ggml_repeat_4d(ctx, bias, kWindowTokens, kWindowTokens, heads, windows);
    mask = ggml_repeat_4d(ctx, mask, kWindowTokens, kWindowTokens, heads, windows);
    return ggml_add(ctx, bias, mask);
}

ggml_tensor * window_attention(ggml_context * ctx, ggml_tensor * x,
                               const weight_store & weights,
                               const std::string & prefix,
                               ggml_tensor * shift_mask, int heads) {
    const int64_t tokens=x->ne[1], windows=x->ne[2], channels=x->ne[0];
    const int64_t head_dim=channels/heads;
    ggml_tensor * qkv=linear(ctx,x,weights,prefix+".qkv");
    auto projection=[&](int part) {
        ggml_tensor * value=ggml_view_3d(
            ctx,qkv,channels,tokens,windows,qkv->nb[1],qkv->nb[2],
            size_t(part)*size_t(channels)*ggml_element_size(qkv));
        value=ggml_cont(ctx,value);
        value=ggml_reshape_4d(ctx,value,head_dim,heads,tokens,windows);
        return ggml_permute(ctx,value,0,2,1,3); // [D,T,H,W]
    };
    ggml_tensor * q=projection(0), * k=projection(1), * v=projection(2);
    k=attention_kv_storage(ctx,k); v=attention_kv_storage(ctx,v);
    ggml_tensor * mask=relative_mask(ctx,weights,prefix,shift_mask,heads,windows);
    ggml_tensor * out=ggml_flash_attn_ext(
        ctx,q,k,v,mask,1.0f/std::sqrt(float(head_dim)),0.0f,0.0f);
    ggml_flash_attn_ext_set_prec(out,GGML_PREC_F32);
    // Flash output is [D,H,T,W]; D and H are adjacent and flatten directly to C.
    out=ggml_reshape_3d(ctx,out,channels,tokens,windows);
    return linear(ctx,out,weights,prefix+".proj");
}

ggml_tensor * swin_block(ggml_context * ctx, ggml_tensor * x,
                         const weight_store & weights, const std::string & prefix,
                         ggml_tensor * shift_mask, int heads,
                         int64_t width, int64_t height, bool shifted) {
    const int64_t channels=x->ne[0], batch=x->ne[2];
    if (batch != 1) throw std::runtime_error("Swin Vulkan window kernels currently require batch=1");
    ggml_tensor * shortcut=x;
    ggml_tensor * h=layer_norm(ctx,x,weights,prefix+".norm1",1e-5f);
    h=ggml_reshape_4d(ctx,h,channels,width,height,batch); // [C,W,H,B]
    const int pad_w=(kWindow-width%kWindow)%kWindow;
    const int pad_h=(kWindow-height%kWindow)%kWindow;
    if (pad_w || pad_h) h=ggml_pad(ctx,h,0,pad_w,pad_h,0);
    const int64_t padded_w=width+pad_w, padded_h=height+pad_h;
    if (shifted) h=ggml_roll(ctx,h,0,-kWindow/2,-kWindow/2,0);
    h=ggml_win_part(ctx,h,kWindow); // [C,WS,WS,num_windows]
    const int64_t windows=h->ne[3];
    h=ggml_reshape_3d(ctx,h,channels,kWindowTokens,windows);
    h=window_attention(ctx,h,weights,prefix+".attn",shifted?shift_mask:nullptr,heads);
    h=ggml_reshape_4d(ctx,h,channels,kWindow,kWindow,windows);
    h=ggml_win_unpart(ctx,h,(int)padded_w,(int)padded_h,kWindow);
    if (shifted) h=ggml_roll(ctx,h,0,kWindow/2,kWindow/2,0);
    if (pad_w || pad_h) {
        h=ggml_view_4d(ctx,h,channels,width,height,1,h->nb[1],h->nb[2],h->nb[3],0);
        h=ggml_cont(ctx,h);
    }
    h=ggml_reshape_3d(ctx,h,channels,width*height,1);
    x=ggml_add(ctx,shortcut,h);
    h=layer_norm(ctx,x,weights,prefix+".norm2",1e-5f);
    h=ggml_gelu(ctx,linear(ctx,h,weights,prefix+".mlp.fc1"));
    h=linear(ctx,h,weights,prefix+".mlp.fc2");
    return ggml_add(ctx,x,h);
}

ggml_tensor * patch_merge(ggml_context * ctx, ggml_tensor * x,
                          const weight_store & weights, const std::string & prefix,
                          int64_t width, int64_t height) {
    const int64_t channels=x->ne[0], batch=x->ne[2];
    ggml_tensor * spatial=ggml_reshape_4d(ctx,x,channels,width,height,batch);
    if (width%2 || height%2) spatial=ggml_pad(ctx,spatial,0,width%2,height%2,0);
    const int64_t out_w=(width+1)/2, out_h=(height+1)/2;
    ggml_tensor * merged=nullptr;
    const int offsets[4][2]={{0,0},{0,1},{1,0},{1,1}};
    for (const auto & offset:offsets) {
        ggml_tensor * part=ggml_view_4d(
            ctx,spatial,channels,out_w,out_h,batch,
            spatial->nb[1]*2,spatial->nb[2]*2,spatial->nb[3],
            size_t(offset[0])*spatial->nb[1]+size_t(offset[1])*spatial->nb[2]);
        part=ggml_cont(ctx,part);
        merged=merged?ggml_concat(ctx,merged,part,0):part;
    }
    merged=ggml_reshape_3d(ctx,merged,4*channels,out_w*out_h,batch);
    merged=layer_norm(ctx,merged,weights,prefix+".norm",1e-5f);
    return linear(ctx,merged,weights,prefix+".reduction");
}

std::array<ggml_tensor *,4> swin_features(ggml_context *ctx,const weight_store&weights,
                                         ggml_tensor*pixels,const std::array<ggml_tensor*,4>&masks){
    pixels=ggml_cast(ctx,pixels,GGML_TYPE_F32);
    ggml_tensor *patch=ggml_conv_2d(ctx,weights.get("bb.patch_embed.proj.weight"),pixels,4,4,0,0,1,1);
    patch=ggml_add(ctx,patch,channel_parameter(ctx,weights.get("bb.patch_embed.proj.bias")));
    int64_t width=patch->ne[0],height=patch->ne[1];
    ggml_tensor*x=layer_norm(ctx,image_to_tokens(ctx,patch),weights,"bb.patch_embed.norm",1e-5f);
    std::array<ggml_tensor*,4> out{};
    for(int stage=0;stage<4;++stage){
        for(int block=0;block<kDepths[stage];++block){
            const std::string p="bb.layers."+std::to_string(stage)+".blocks."+std::to_string(block);
            x=swin_block(ctx,x,weights,p,masks[stage],kHeads[stage],width,height,(block%2)==1);
        }
        ggml_tensor*n=layer_norm(ctx,x,weights,"bb.norm"+std::to_string(stage),1e-5f);
        out[stage]=tokens_to_image(ctx,n,width,height);
        if(stage<3){x=patch_merge(ctx,x,weights,"bb.layers."+std::to_string(stage)+".downsample",width,height);width=(width+1)/2;height=(height+1)/2;}
    }
    return out;
}

ggml_tensor * conv2d(ggml_context*ctx,ggml_tensor*x,const weight_store&w,const std::string&p,
                     int stride=1,int padding=0,bool bias=true){
    ggml_tensor*y=ggml_conv_2d(ctx,w.get(p+".weight"),x,stride,stride,padding,padding,1,1);
    if(bias&&w.maybe(p+".bias"))y=ggml_add(ctx,y,channel_parameter(ctx,w.get(p+".bias")));
    return y;
}

ggml_tensor * batch_norm(ggml_context*ctx,ggml_tensor*x,const weight_store&w,const std::string&p){
    ggml_tensor*mean=channel_parameter(ctx,w.get(p+".running_mean"));
    ggml_tensor*var=channel_parameter(ctx,w.get(p+".running_var"));
    ggml_tensor*scale=channel_parameter(ctx,w.get(p+".weight"));
    ggml_tensor*bias=channel_parameter(ctx,w.get(p+".bias"));
    ggml_tensor*eps=ggml_fill(ctx,var,1e-5f);
    return ggml_add(ctx,ggml_mul(ctx,ggml_div(ctx,ggml_sub(ctx,x,mean),
        ggml_sqrt(ctx,ggml_add(ctx,var,eps))),scale),bias);
}

ggml_tensor * interpolate(ggml_context*ctx,ggml_tensor*x,int64_t width,int64_t height){
    return ggml_interpolate(ctx,x,width,height,x->ne[2],x->ne[3],
        GGML_SCALE_MODE_BILINEAR|GGML_SCALE_FLAG_ALIGN_CORNERS);
}

ggml_tensor * concat_channels(ggml_context*ctx,const std::vector<ggml_tensor*>&xs){
    ggml_tensor*out=xs.at(0);for(size_t i=1;i<xs.size();++i)out=ggml_concat(ctx,out,xs[i],2);return out;
}

ggml_tensor * global_mean(ggml_context*ctx,ggml_tensor*x){
    const int64_t w=x->ne[0],h=x->ne[1],c=x->ne[2],n=x->ne[3];
    x=ggml_reshape_2d(ctx,x,w*h,c*n);x=ggml_mean(ctx,x);return ggml_reshape_4d(ctx,x,1,1,c,n);
}

ggml_tensor * deform_branch(ggml_context*ctx,ggml_tensor*x,const weight_store&w,
                            const std::string&p,int kernel){
    const int pad=kernel/2;
    ggml_tensor*offset=conv2d(ctx,x,w,p+".atrous_conv.offset_conv",1,pad,true);
    ggml_tensor*mask=conv2d(ctx,x,w,p+".atrous_conv.modulator_conv",1,pad,true);
    mask=ggml_scale(ctx,ggml_sigmoid(ctx,mask),2.0f);
    ggml_tensor*y=deform_conv_2d(ctx,x,offset,mask,w.get(p+".atrous_conv.regular_conv.weight"),1,1,pad,pad);
    return ggml_relu(ctx,batch_norm(ctx,y,w,p+".bn"));
}

ggml_tensor * aspp(ggml_context*ctx,ggml_tensor*x,const weight_store&w,const std::string&p){
    std::vector<ggml_tensor*> branches;
    branches.push_back(deform_branch(ctx,x,w,p+".aspp1",1));
    branches.push_back(deform_branch(ctx,x,w,p+".aspp_deforms.0",1));
    branches.push_back(deform_branch(ctx,x,w,p+".aspp_deforms.1",3));
    branches.push_back(deform_branch(ctx,x,w,p+".aspp_deforms.2",7));
    ggml_tensor*g=global_mean(ctx,x);g=conv2d(ctx,g,w,p+".global_avg_pool.1",1,0,false);
    g=ggml_relu(ctx,batch_norm(ctx,g,w,p+".global_avg_pool.2"));
    branches.push_back(interpolate(ctx,g,x->ne[0],x->ne[1]));
    ggml_tensor*y=conv2d(ctx,concat_channels(ctx,branches),w,p+".conv1",1,0,false);
    return ggml_relu(ctx,batch_norm(ctx,y,w,p+".bn1"));
}

ggml_tensor * basic_dec(ggml_context*ctx,ggml_tensor*x,const weight_store&w,const std::string&p){
    x=ggml_relu(ctx,batch_norm(ctx,conv2d(ctx,x,w,p+".conv_in",1,1),w,p+".bn_in"));
    x=aspp(ctx,x,w,p+".dec_att");
    return batch_norm(ctx,conv2d(ctx,x,w,p+".conv_out",1,1),w,p+".bn_out");
}

ggml_tensor * simple_conv(ggml_context*ctx,ggml_tensor*x,const weight_store&w,const std::string&p){
    return conv2d(ctx,conv2d(ctx,x,w,p+".conv1",1,1),w,p+".conv_out",1,1);
}

ggml_tensor * gated(ggml_context*ctx,ggml_tensor*x,const weight_store&w,int level){
    const std::string base="decoder.gdt_convs_"+std::to_string(level);
    ggml_tensor*g=conv2d(ctx,x,w,base+".0",1,1);
    g=ggml_relu(ctx,batch_norm(ctx,g,w,base+".1"));
    g=ggml_sigmoid(ctx,conv2d(ctx,g,w,"decoder.gdt_convs_attn_"+std::to_string(level)+".0"));
    return ggml_mul(ctx,x,g);
}

} // namespace

ggml_tensor * build_swin_backbone(ggml_context * ctx,
                                  const weight_store & weights,
                                  const swin_inputs & in,
                                  int stop_stage,
                                  int blocks_in_stop_stage) {
    if (stop_stage < 0 || stop_stage > 3) throw std::runtime_error("invalid Swin stage");
    ggml_tensor * pixels=ggml_cast(ctx,in.pixels,GGML_TYPE_F32);
    ggml_tensor * patch=ggml_conv_2d(
        ctx,weights.get("bb.patch_embed.proj.weight"),pixels,4,4,0,0,1,1);
    patch=ggml_add(ctx,patch,channel_parameter(ctx,weights.get("bb.patch_embed.proj.bias")));
    int64_t width=patch->ne[0],height=patch->ne[1];
    ggml_tensor * x=image_to_tokens(ctx,patch);
    x=layer_norm(ctx,x,weights,"bb.patch_embed.norm",1e-5f);

    for(int stage=0;stage<=stop_stage;++stage){
        const int block_count=(stage==stop_stage && blocks_in_stop_stage>=0)
            ? std::min(blocks_in_stop_stage,kDepths[stage]):kDepths[stage];
        for(int block=0;block<block_count;++block){
            const std::string prefix="bb.layers."+std::to_string(stage)+".blocks."+std::to_string(block);
            x=swin_block(ctx,x,weights,prefix,in.masks[stage],kHeads[stage],width,height,(block%2)==1);
        }
        if(stage==stop_stage){
            x=layer_norm(ctx,x,weights,"bb.norm"+std::to_string(stage),1e-5f);
            return tokens_to_image(ctx,x,width,height);
        }
        x=patch_merge(ctx,x,weights,"bb.layers."+std::to_string(stage)+".downsample",width,height);
        width=(width+1)/2;height=(height+1)/2;
    }
    throw std::runtime_error("unreachable Swin graph state");
}

ggml_tensor * build_birefnet(ggml_context *ctx,const weight_store&weights,const biref_inputs&in){
    ggml_tensor*pixels=ggml_cast(ctx,in.pixels,GGML_TYPE_F32);
    auto full=swin_features(ctx,weights,pixels,in.full_masks);
    ggml_tensor*half_pixels=interpolate(ctx,pixels,pixels->ne[0]/2,pixels->ne[1]/2);
    auto half=swin_features(ctx,weights,half_pixels,in.half_masks);
    std::array<ggml_tensor*,4>x{};
    for(int i=0;i<4;++i){
        ggml_tensor*h=interpolate(ctx,half[i],full[i]->ne[0],full[i]->ne[1]);
        x[i]=ggml_concat(ctx,full[i],h,2);
    }
    ggml_tensor*x4=concat_channels(ctx,{
        interpolate(ctx,x[0],x[3]->ne[0],x[3]->ne[1]),
        interpolate(ctx,x[1],x[3]->ne[0],x[3]->ne[1]),
        interpolate(ctx,x[2],x[3]->ne[0],x[3]->ne[1]),x[3]});
    x4=basic_dec(ctx,x4,weights,"squeeze_module.0");

    ggml_tensor*ipt=image_to_patches(ctx,pixels,x4->ne[0],x4->ne[1]);
    x4=concat_channels(ctx,{x4,simple_conv(ctx,ipt,weights,"decoder.ipt_blk5")});
    ggml_tensor*p4=gated(ctx,basic_dec(ctx,x4,weights,"decoder.decoder_block4"),weights,4);
    ggml_tensor*p3=ggml_add(ctx,interpolate(ctx,p4,x[2]->ne[0],x[2]->ne[1]),
        conv2d(ctx,x[2],weights,"decoder.lateral_block4.conv"));
    ipt=image_to_patches(ctx,pixels,p3->ne[0],p3->ne[1]);
    p3=concat_channels(ctx,{p3,simple_conv(ctx,ipt,weights,"decoder.ipt_blk4")});
    p3=gated(ctx,basic_dec(ctx,p3,weights,"decoder.decoder_block3"),weights,3);

    ggml_tensor*p2=ggml_add(ctx,interpolate(ctx,p3,x[1]->ne[0],x[1]->ne[1]),
        conv2d(ctx,x[1],weights,"decoder.lateral_block3.conv"));
    ipt=image_to_patches(ctx,pixels,p2->ne[0],p2->ne[1]);
    p2=concat_channels(ctx,{p2,simple_conv(ctx,ipt,weights,"decoder.ipt_blk3")});
    p2=gated(ctx,basic_dec(ctx,p2,weights,"decoder.decoder_block2"),weights,2);

    ggml_tensor*p1=ggml_add(ctx,interpolate(ctx,p2,x[0]->ne[0],x[0]->ne[1]),
        conv2d(ctx,x[0],weights,"decoder.lateral_block2.conv"));
    ipt=image_to_patches(ctx,pixels,p1->ne[0],p1->ne[1]);
    p1=concat_channels(ctx,{p1,simple_conv(ctx,ipt,weights,"decoder.ipt_blk2")});
    p1=basic_dec(ctx,p1,weights,"decoder.decoder_block1");
    p1=interpolate(ctx,p1,pixels->ne[0],pixels->ne[1]);
    ipt=image_to_patches(ctx,pixels,p1->ne[0],p1->ne[1]);
    p1=concat_channels(ctx,{p1,simple_conv(ctx,ipt,weights,"decoder.ipt_blk1")});
    return ggml_sigmoid(ctx,conv2d(ctx,p1,weights,"decoder.conv_out1.0"));
}

} // namespace triposplat
