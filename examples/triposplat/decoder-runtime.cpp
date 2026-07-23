#include "decoder-runtime.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace triposplat {
namespace {

constexpr float kC0 = 0.28209479177387814f;
constexpr std::array<std::array<int, 3>, 8> kChildOffsets {{
    {{0,0,0}}, {{1,0,0}}, {{0,1,0}}, {{1,1,0}},
    {{0,0,1}}, {{1,0,1}}, {{0,1,1}}, {{1,1,1}},
}};
constexpr std::array<std::array<float, 3>, 3> kDefaultTransform {{
    {{1,0,0}}, {{0,0,-1}}, {{0,1,0}},
}};

float softplus(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

float logit(float x) { return std::log(x / (1.0f - x)); }

float inverse_softplus(float x) { return std::log(std::expm1(x)); }

float radical_inverse(int base, size_t n) {
    float value=0.0f, inverse=1.0f/float(base), factor=inverse;
    while(n){ value += float(n%size_t(base))*factor; n/=size_t(base); factor*=inverse; }
    return value;
}

std::array<float,3> hammersley3(size_t n, size_t count) {
    return {float(n)/float(count), radical_inverse(2,n), radical_inverse(3,n)};
}

void append(std::vector<uint8_t> & out, const void * data, size_t bytes) {
    const auto * p=static_cast<const uint8_t *>(data);
    out.insert(out.end(),p,p+bytes);
}

void append_f32(std::vector<uint8_t> & out, float value) { append(out,&value,sizeof(value)); }

std::array<float,4> normalize_quat(std::array<float,4> q) {
    float n=std::sqrt(std::inner_product(q.begin(),q.end(),q.begin(),0.0f));
    if(!(n>0)) return {1,0,0,0};
    for(float & v:q) v/=n;
    return q;
}

std::array<std::array<float,3>,3> quat_to_matrix(std::array<float,4> q) {
    q=normalize_quat(q); const float w=q[0],x=q[1],y=q[2],z=q[3];
    return {{{{1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)}},
             {{2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)}},
             {{2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)}}}};
}

std::array<float,4> matrix_to_quat(const std::array<std::array<float,3>,3> & m) {
    std::array<float,4> q{};
    float trace=m[0][0]+m[1][1]+m[2][2];
    if(trace>0){ float s=std::sqrt(trace+1)*2; q={.25f*s,(m[2][1]-m[1][2])/s,(m[0][2]-m[2][0])/s,(m[1][0]-m[0][1])/s}; }
    else if(m[0][0]>=m[1][1]&&m[0][0]>=m[2][2]){ float s=std::sqrt(std::max(0.0f,1+m[0][0]-m[1][1]-m[2][2]))*2; q={(m[2][1]-m[1][2])/s,.25f*s,(m[0][1]+m[1][0])/s,(m[0][2]+m[2][0])/s}; }
    else if(m[1][1]>=m[2][2]){ float s=std::sqrt(std::max(0.0f,1+m[1][1]-m[0][0]-m[2][2]))*2; q={(m[0][2]-m[2][0])/s,(m[0][1]+m[1][0])/s,.25f*s,(m[1][2]+m[2][1])/s}; }
    else { float s=std::sqrt(std::max(0.0f,1+m[2][2]-m[0][0]-m[1][1]))*2; q={(m[1][0]-m[0][1])/s,(m[0][2]+m[2][0])/s,(m[1][2]+m[2][1])/s,.25f*s}; }
    return normalize_quat(q);
}

std::array<float,3> transform_xyz(const float * p) {
    std::array<float,3> r{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) r[i]+=kDefaultTransform[i][j]*p[j];
    return r;
}

std::array<float,4> transform_rotation(const float * raw) {
    auto m=quat_to_matrix({1+raw[0],raw[1],raw[2],raw[3]});
    std::array<std::array<float,3>,3> r{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) r[i][j]+=kDefaultTransform[i][k]*m[k][j];
    return matrix_to_quat(r);
}

uint8_t quantize(float x) { return uint8_t(std::clamp(x,0.0f,255.0f)); }

} // namespace

decoder_point_set sample_octree(const octree_logits_fn & infer,size_t num_points,int max_level,float temperature,uint64_t seed) {
    if(!infer||!num_points||max_level<1||max_level>30||!(temperature>0)) throw std::invalid_argument("invalid octree sampling arguments");
    struct cell { std::array<int32_t,3> xyz; size_t count; float log_prob; };
    std::vector<cell> cells {{{{0,0,0}},num_points,0.0f}};
    std::mt19937_64 rng(seed); std::uniform_real_distribution<float> uniform(0.0f,1.0f);
    for(int level=1;level<=max_level;++level){
        const int parent_res=1<<(level-1), resolution=1<<level;
        std::vector<float> p(cells.size()*3);
        for(size_t i=0;i<cells.size();++i) for(int d=0;d<3;++d) p[3*i+d]=(float(cells[i].xyz[d])+.5f)/float(parent_res);
        std::vector<float> logits=infer(p,resolution);
        if(logits.size()!=cells.size()*8) throw std::runtime_error("octree callback returned incorrect logits size");
        std::vector<cell> next; next.reserve(std::min(num_points,cells.size()*8));
        for(size_t i=0;i<cells.size();++i){
            std::array<float,8> prob{},log_prob{}; float mx=-std::numeric_limits<float>::infinity();
            for(int j=0;j<8;++j) mx=std::max(mx,logits[i*8+j]/temperature);
            float sum=0; for(int j=0;j<8;++j) sum+=prob[j]=std::exp(logits[i*8+j]/temperature-mx);
            if(!(sum>0)){ prob.fill(1.0f/8); sum=1; }
            float accum=0; for(int j=0;j<8;++j){ prob[j]/=sum; log_prob[j]=std::log(std::max(prob[j],std::numeric_limits<float>::min())); }
            // One random systematic offset per parent, exactly as sample_probs.
            const float u0=uniform(rng)/float(cells[i].count);
            std::array<size_t,8> counts{}; int bin=0; accum=prob[0];
            for(size_t sample=0;sample<cells[i].count;++sample){
                float u=std::min(u0+float(sample)/float(cells[i].count),std::nextafter(1.0f,0.0f));
                while(bin<7&&u>=accum) accum+=prob[++bin];
                ++counts[bin];
            }
            for(int j=0;j<8;++j) if(counts[j]){
                cell c{}; c.count=counts[j]; c.log_prob=cells[i].log_prob+log_prob[j];
                for(int d=0;d<3;++d)c.xyz[d]=cells[i].xyz[d]*2+kChildOffsets[j][d];
                next.push_back(c);
            }
        }
        cells=std::move(next);
    }
    decoder_point_set result; result.points.reserve(num_points*3); result.log_probs.reserve(num_points);
    const float resolution=float(1<<max_level);
    for(const cell & c:cells) for(size_t n=0;n<c.count;++n){
        for(int d=0;d<3;++d) result.points.push_back((float(c.xyz[d])+uniform(rng))/resolution);
        result.log_probs.push_back(c.log_prob);
    }
    if(result.size()!=num_points) throw std::runtime_error("octree sampler did not preserve point count");
    return result;
}

gaussian_cloud build_gaussians(const decoder_point_set & points,const std::vector<float> & h,const gaussian_config & c) {
    const size_t ng=c.gaussians_per_point, channels=ng*(3+3+3+4+1+(c.learned_offset_scale?1:0));
    if(points.points.size()%3||h.size()!=points.size()*channels) throw std::invalid_argument("incorrect GS decoder feature shape");
    gaussian_cloud g; const size_t total=points.size()*ng;
    g.xyz.resize(total*3); g.features.resize(total*3); g.opacity.resize(total);
    g.opacity_logit.resize(total); g.scaling.resize(total*3); g.rotation.resize(total*4);
    const size_t xyz0=0,dc0=3*ng,scale0=6*ng,rot0=9*ng,opacity0=13*ng,offset0=14*ng;
    const float base_offset=inverse_softplus(c.offset_scale), scale_bias=inverse_softplus(c.scaling_bias), opacity_bias=logit(c.opacity_bias);
    for(size_t p=0;p<points.size();++p) for(size_t j=0;j<ng;++j){
        const size_t out=p*ng+j, base=p*channels; auto perturb=hammersley3(j,ng);
        float offset_scale=c.learned_offset_scale?softplus(h[base+offset0+j]+base_offset):c.offset_scale;
        for(int d=0;d<3;++d){
            float raw=h[base+xyz0+j*3+d];
            if(c.perturb_offset) raw+=std::atanh((perturb[d]*2-1)/c.perturb_size);
            float offset=std::tanh(raw)*.5f*c.perturb_size*offset_scale;
            g.xyz[out*3+d]=points.points[p*3+d]+offset-.5f;
            g.features[out*3+d]=h[base+dc0+j*3+d];
            float s=softplus(h[base+scale0+j*3+d]+scale_bias);
            g.scaling[out*3+d]=std::sqrt(s*s+c.minimum_kernel_size*c.minimum_kernel_size);
        }
        for(int d=0;d<4;++d) g.rotation[out*4+d]=h[base+rot0+j*4+d]*c.rotation_lr;
        g.opacity_logit[out]=h[base+opacity0+j]+opacity_bias;
        g.opacity[out]=1/(1+std::exp(-g.opacity_logit[out]));
    }
    return g;
}

std::vector<uint8_t> gaussian_to_ply(const gaussian_cloud & g) {
    if(g.xyz.size()!=g.size()*3||g.features.size()!=g.size()*3||g.opacity_logit.size()!=g.size()||g.scaling.size()!=g.size()*3||g.rotation.size()!=g.size()*4) throw std::invalid_argument("invalid Gaussian cloud");
    std::string header="ply\nformat binary_little_endian 1.0\nelement vertex "+std::to_string(g.size())+"\n";
    for(const char * name:{"x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"}) header += std::string("property float ")+name+"\n";
    header+="end_header\n"; std::vector<uint8_t> out(header.begin(),header.end()); out.reserve(out.size()+g.size()*17*4);
    for(size_t i=0;i<g.size();++i){
        auto xyz=transform_xyz(&g.xyz[i*3]); for(float v:xyz)append_f32(out,v); for(int d=0;d<3;++d)append_f32(out,0);
        for(int d=0;d<3;++d) append_f32(out,g.features[i*3+d]);
        append_f32(out,g.opacity_logit[i]);
        for(int d=0;d<3;++d) append_f32(out,std::log(g.scaling[i*3+d]));
        auto q=transform_rotation(&g.rotation[i*4]);
        for(float v:q) append_f32(out,v);
    }
    return out;
}

std::vector<uint8_t> gaussian_to_splat(const gaussian_cloud & g) {
    std::vector<size_t> order(g.size()); std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](size_t a,size_t b){ return g.opacity[a]*g.scaling[a*3]*g.scaling[a*3+1]*g.scaling[a*3+2] > g.opacity[b]*g.scaling[b*3]*g.scaling[b*3+1]*g.scaling[b*3+2]; });
    std::vector<uint8_t> out; out.reserve(g.size()*32);
    for(size_t i:order){ auto xyz=transform_xyz(&g.xyz[i*3]); for(float v:xyz)append_f32(out,v); for(int d=0;d<3;++d)append_f32(out,g.scaling[i*3+d]);
        for(int d=0;d<3;++d) out.push_back(quantize((g.features[i*3+d]*kC0+.5f)*255));
        out.push_back(quantize(g.opacity[i]*255));
        auto q=transform_rotation(&g.rotation[i*4]); for(float v:q)out.push_back(quantize(v*128+128)); }
    return out;
}

void save_binary_file(const std::string & path,const std::vector<uint8_t> & data) {
    std::ofstream f(path,std::ios::binary); if(!f)throw std::runtime_error("cannot open output file: "+path);
    f.write(reinterpret_cast<const char *>(data.data()),std::streamsize(data.size())); if(!f)throw std::runtime_error("cannot write output file: "+path);
}

} // namespace triposplat
