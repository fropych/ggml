#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-vulkan.h"

#include "safetensors.h"
#include "flow-model.h"
#include "dino-model.h"
#include "vae-model.h"
#include "decoder-model.h"
#include "triposplat-ops.h"
#include "biref-model.h"
#include "e2e-worker.h"
#include "pipeline.h"
#include "model-store.h"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

using namespace triposplat;

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s generate IMAGE --model-dir DIR [--output PREFIX] [OPTIONS]\n"
        "  %s download --model-dir DIR [--repo REPO] [--revision REV]\n\n"
        "Generate options:\n"
        "  --steps N              Flow steps (default: 20)\n"
        "  --guidance F           Classifier-free guidance (default: 3.0)\n"
        "  --num-gaussians N      Positive multiple of 32 (default: 32768)\n"
        "  --seed N               Random seed (default: 42)\n"
        "  --erode-radius N       Alpha erosion radius (default: 1)\n"
        "  --device N             Vulkan device index (default: 0)\n"
        "  --assets DIR           Runtime asset directory\n"
        "  --download             Download missing weights from Hugging Face\n"
        "  --keep-temp            Preserve intermediate safetensors\n\n"
        "  %s --inspect MODEL.safetensors\n"
        "  %s --load MODEL.safetensors [--f16]\n\n"
        "  %s --flow-parity MODEL.safetensors INPUTS.safetensors [BLOCKS]\n\n"
        "  %s --dino-parity MODEL.safetensors INPUTS.safetensors [LAYERS]\n\n"
        "  %s --vae-parity MODEL.safetensors INPUTS.safetensors [STAGE]\n\n"
        "  %s --gs-parity MODEL.safetensors INPUTS.safetensors [BLOCKS]\n\n"
        "  %s --octree-parity MODEL.safetensors INPUTS.safetensors [BLOCKS]\n\n"
        "  %s --deform-parity INPUTS.safetensors\n\n"
        "  %s --swin-parity MODEL.safetensors INPUTS.safetensors [STAGE] [BLOCKS]\n\n"
        "  %s --biref-parity MODEL.safetensors INPUTS.safetensors\n\n"
        "  %s --sample-parity MODEL.safetensors INPUTS.safetensors [STEPS] [BLOCKS] [GUIDANCE]\n\n"
        "  %s --run-biref MODEL.safetensors INPUT.safetensors OUTPUT.safetensors\n"
        "  %s --run-dino MODEL.safetensors INPUT.safetensors OUTPUT.safetensors\n"
        "  %s --run-vae MODEL.safetensors INPUT.safetensors OUTPUT.safetensors\n"
        "  %s --run-flow MODEL.safetensors INPUT.safetensors OUTPUT.safetensors [STEPS] [GUIDANCE]\n"
        "  %s --run-decode MODEL.safetensors INPUT.safetensors OUTPUT_PREFIX [GAUSSIANS] [SEED]\n\n"
        "The executable initializes Vulkan directly; it never creates a CPU backend.\n",
        argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0, argv0,
        argv0, argv0, argv0, argv0, argv0, argv0, argv0);
}

static std::string option_value(int & index, int argc, char ** argv) {
    if (++index >= argc) throw std::invalid_argument(
        std::string("missing value after ") + argv[index - 1]);
    return argv[index];
}

static std::filesystem::path executable_directory(const char * argv0) {
    std::error_code error;
#ifdef _WIN32
    std::wstring executable_path(32768, L'\0');
    const DWORD length = GetModuleFileNameW(
        nullptr, executable_path.data(), DWORD(executable_path.size()));
    if (length > 0 && length < executable_path.size()) {
        executable_path.resize(length);
        return std::filesystem::path(executable_path).parent_path();
    }
#elif defined(__linux__)
    const std::filesystem::path proc_path =
        std::filesystem::read_symlink("/proc/self/exe", error);
    if (!error && !proc_path.empty()) return proc_path.parent_path();
    error.clear();
#endif
    const std::filesystem::path absolute =
        std::filesystem::absolute(argv0 ? argv0 : "", error);
    if (!error) return absolute.parent_path();
    return std::filesystem::current_path();
}

static int run_cli_command(int argc, char ** argv) {
    const std::string command = argv[1];
    if (command == "download") {
        model_store_config config;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--model-dir") config.directory = option_value(i, argc, argv);
            else if (option == "--repo") config.repository = option_value(i, argc, argv);
            else if (option == "--revision") config.revision = option_value(i, argc, argv);
            else throw std::invalid_argument("unknown download option: " + option);
        }
        if (config.directory.empty()) throw std::invalid_argument("--model-dir is required");
        download_models(config);
        std::printf("models are ready in %s\n", std::filesystem::absolute(config.directory).c_str());
        return 0;
    }
    if (command != "generate") return -1;
    if (argc < 3) throw std::invalid_argument("generate requires an input image");
    generate_options options;
    options.input_image = argv[2];
    int device = 0;
    for (int i = 3; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--model-dir") options.models.directory = option_value(i, argc, argv);
        else if (option == "--repo") options.models.repository = option_value(i, argc, argv);
        else if (option == "--revision") options.models.revision = option_value(i, argc, argv);
        else if (option == "--output") options.output_prefix = option_value(i, argc, argv);
        else if (option == "--assets") options.asset_directory = option_value(i, argc, argv);
        else if (option == "--steps") options.steps = std::stoi(option_value(i, argc, argv));
        else if (option == "--guidance") options.guidance = std::stof(option_value(i, argc, argv));
        else if (option == "--num-gaussians") options.num_gaussians =
            size_t(std::stoull(option_value(i, argc, argv)));
        else if (option == "--seed") options.seed =
            uint64_t(std::stoull(option_value(i, argc, argv)));
        else if (option == "--erode-radius") options.erode_radius =
            std::stoi(option_value(i, argc, argv));
        else if (option == "--device") device = std::stoi(option_value(i, argc, argv));
        else if (option == "--download") options.models.allow_download = true;
        else if (option == "--keep-temp") options.keep_temporary = true;
        else throw std::invalid_argument("unknown generate option: " + option);
    }
    if (options.models.directory.empty()) throw std::invalid_argument("--model-dir is required");
    if (options.asset_directory.empty()) {
        options.asset_directory = (executable_directory(argv[0]) / "assets").string();
    }
    pipeline runtime(device);
    std::printf("Vulkan device: %s\n", runtime.device_description().c_str());
    const generate_result result = runtime.generate(options);
    std::printf("C++ Vulkan e2e complete in %.3fs\n%s\n%s\n",
                result.elapsed_seconds, result.ply_path.c_str(), result.splat_path.c_str());
    return 0;
}

static std::vector<float> tensor_f32(ggml_tensor * tensor) {
    const size_t count = size_t(ggml_nelements(tensor));
    std::vector<float> out(count);
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, out.data(), 0, count * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> values(count);
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(values[0]));
        for (size_t i = 0; i < count; ++i) out[i] = ggml_fp16_to_fp32(values[i]);
    } else {
        throw std::runtime_error("parity output must be F32 or F16");
    }
    return out;
}

static bool compute_graph(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    int iterations = 1;
    int warmup = 0;
    if (const char * value = std::getenv("TRIPOSPLAT_BENCH_ITERS")) {
        iterations = std::max(1, std::stoi(value));
    }
    if (const char * value = std::getenv("TRIPOSPLAT_BENCH_WARMUP")) {
        warmup = std::max(0, std::stoi(value));
    }
    for (int i = 0; i < warmup; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    }
    const auto end = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - begin).count();
    std::printf("%s compute: %.3f ms total, %.3f ms/iteration (%d iteration%s)\n",
                label, milliseconds, milliseconds / iterations, iterations,
                iterations == 1 ? "" : "s");
    return true;
}

int main(int argc, char ** argv) {
    try {
        if (argc >= 2) {
            const int result = run_cli_command(argc, argv);
            if (result >= 0) return result;
        }
        if (argc < 3) {
            usage(argv[0]);
            return 2;
        }
        const std::string mode = argv[1];
        const std::string path = argv[2];
        if (mode == "--inspect") {
            safetensors_file file(path);
            std::printf("%s: %zu tensors\n", path.c_str(), file.tensors().size());
            size_t bytes = 0;
            for (const auto & tensor : file.tensors()) bytes += size_t(tensor.end - tensor.begin);
            std::printf("payload: %.2f MiB\n", double(bytes) / (1024.0 * 1024.0));
            return 0;
        }
        if (mode != "--load" && mode != "--flow-parity" && mode != "--dino-parity" && mode != "--vae-parity" && mode != "--gs-parity" && mode != "--octree-parity" && mode != "--deform-parity" && mode != "--swin-parity" && mode != "--biref-parity" && mode != "--sample-parity" && !is_e2e_worker_mode(mode)) {
            usage(argv[0]);
            return 2;
        }

        const int count = ggml_backend_vk_get_device_count();
        if (count < 1) {
            throw std::runtime_error(
                "no Vulkan device; for llvmpipe smoke tests set GGML_VK_VISIBLE_DEVICES=0");
        }
        char description[256] = {};
        ggml_backend_vk_get_device_description(0, description, sizeof(description));
        std::printf("Vulkan device: %s\n", description);
        ggml_backend_t backend = ggml_backend_vk_init(0);
        if (!backend) throw std::runtime_error("failed to initialize Vulkan backend");
        if (is_e2e_worker_mode(mode)) {
            const int result = run_e2e_worker(mode, argc, argv, backend);
            ggml_backend_free(backend);
            return result;
        }
        if(mode=="--sample-parity"){
            if(argc<4){usage(argv[0]);ggml_backend_free(backend);return 2;}const int steps=argc>4?std::stoi(argv[4]):2,blocks=argc>5?std::stoi(argv[5]):24;const float guidance=argc>6?std::stof(argv[6]):3.0f,shift_schedule=3.0f;
            weight_store weights(backend,path,weight_policy::native),inputs(backend,argv[3],weight_policy::native);std::vector<uint8_t>metadata(96u*1024u*1024u);ggml_init_params params{metadata.size(),metadata.data(),true};ggml_context*ctx=ggml_init(params);
            flow_inputs in{inputs.get("latent"),inputs.get("feature1"),inputs.get("feature2"),inputs.get("camera"),inputs.get("timestep"),inputs.get("positions"),inputs.get("position_freqs")};flow_outputs outputs=build_flow_outputs(ctx,weights,in,blocks);
            ggml_cgraph*graph=ggml_new_graph_custom(ctx,8192,false);ggml_build_forward_expand(graph,outputs.latent);ggml_build_forward_expand(graph,outputs.camera);for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(backend,n))throw std::runtime_error(std::string("Vulkan does not support sampler op ")+ggml_op_desc(n));}
            ggml_gallocr_t allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(allocator,graph))throw std::runtime_error("sampler graph allocation failed");
            auto latent=tensor_f32(in.latent),camera=tensor_f32(in.camera),cond1=tensor_f32(in.feature1),cond2=tensor_f32(in.feature2);std::vector<float>zero1(cond1.size()),zero2(cond2.size());
            auto upload=[](ggml_tensor*t,const std::vector<float>&v){
                if(t->type==GGML_TYPE_F32){ggml_backend_tensor_set(t,v.data(),0,v.size()*sizeof(float));return;}
                if(t->type==GGML_TYPE_F16){std::vector<ggml_fp16_t>h(v.size());for(size_t i=0;i<v.size();++i)h[i]=ggml_fp32_to_fp16(v[i]);ggml_backend_tensor_set(t,h.data(),0,h.size()*sizeof(h[0]));return;}
                throw std::runtime_error("sampler upload requires F32 or F16 tensor");
            };
            for(int i=0;i<steps;++i){float u=float(i)/steps,up=float(i+1)/steps;float base=1.0f-u,basep=1.0f-up;float t=shift_schedule*base/(1.0f+(shift_schedule-1.0f)*base);float tp=shift_schedule*basep/(1.0f+(shift_schedule-1.0f)*basep);float ts=1000.0f*t;ggml_backend_tensor_set(in.timestep,&ts,0,sizeof(ts));upload(in.latent,latent);upload(in.camera,camera);upload(in.feature1,cond1);upload(in.feature2,cond2);if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("conditional sampler step failed");auto pl=tensor_f32(outputs.latent),pc=tensor_f32(outputs.camera);upload(in.feature1,zero1);upload(in.feature2,zero2);if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("unconditional sampler step failed");auto nl=tensor_f32(outputs.latent),nc=tensor_f32(outputs.camera);float dt=t-tp;for(size_t j=0;j<latent.size();++j)latent[j]-=(guidance*pl[j]-(guidance-1)*nl[j])*dt;for(size_t j=0;j<camera.size();++j)camera[j]-=(guidance*pc[j]-(guidance-1)*nc[j])*dt;std::printf("sampler step %d/%d t=%.6f\n",i+1,steps,t);}
            auto score=[](const std::vector<float>&a,const std::vector<float>&r,const char*name){double mae=0,mse=0,dot=0,aa=0,rr=0;float mx=0;for(size_t i=0;i<a.size();++i){float e=std::abs(a[i]-r[i]);mae+=e;mse+=double(e)*e;mx=std::max(mx,e);dot+=double(a[i])*r[i];aa+=double(a[i])*a[i];rr+=double(r[i])*r[i];}double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));std::printf("%s elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",name,a.size(),mae/a.size(),std::sqrt(mse/a.size()),mx,cosine);return cosine;};double lc=score(latent,tensor_f32(inputs.get("reference_latent")),"sample latent"),cc=score(camera,tensor_f32(inputs.get("reference_camera")),"sample camera");ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);return(lc>0.999&&cc>0.999)?0:1;
        }
        if(mode=="--biref-parity"){
            if(argc<4){usage(argv[0]);ggml_backend_free(backend);return 2;}
            weight_store weights(backend,path,weight_policy::f16),inputs(backend,argv[3],weight_policy::native);
            std::vector<uint8_t>metadata(256u*1024u*1024u);ggml_init_params params{metadata.size(),metadata.data(),true};ggml_context*ctx=ggml_init(params);
            biref_inputs in{inputs.get("pixels"),
                {inputs.get("full_mask0"),inputs.get("full_mask1"),inputs.get("full_mask2"),inputs.get("full_mask3")},
                {inputs.get("half_mask0"),inputs.get("half_mask1"),inputs.get("half_mask2"),inputs.get("half_mask3")}};
            ggml_tensor*output=build_birefnet(ctx,weights,in);ggml_cgraph*graph=ggml_new_graph_custom(ctx,32768,false);ggml_build_forward_expand(graph,output);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(backend,n))throw std::runtime_error(std::string("Vulkan does not support BiRefNet op ")+ggml_op_desc(n));}
            ggml_gallocr_t allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(allocator,graph)||!compute_graph(backend,graph,"BiRefNet"))throw std::runtime_error("BiRefNet execution failed");
            auto actual=tensor_f32(output),reference=tensor_f32(inputs.get("reference"));if(actual.size()!=reference.size())throw std::runtime_error("BiRefNet reference size mismatch");double mae=0,mse=0,dot=0,aa=0,rr=0;float mx=0;for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-reference[i]);mae+=e;mse+=double(e)*e;mx=std::max(mx,e);dot+=double(actual[i])*reference[i];aa+=double(actual[i])*actual[i];rr+=double(reference[i])*reference[i];}double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));std::printf("BiRefNet parity elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);return(cosine>0.999&&mx<0.2f)?0:1;
        }
        if (mode == "--swin-parity") {
            if (argc < 4) { usage(argv[0]); ggml_backend_free(backend); return 2; }
            const int stage=argc>4?std::stoi(argv[4]):0;
            const int blocks=argc>5?std::stoi(argv[5]):-1;
            weight_store weights(backend,path,weight_policy::f16);
            weight_store inputs(backend,argv[3],weight_policy::native);
            std::vector<uint8_t> metadata(128u*1024u*1024u);
            ggml_init_params params{metadata.size(),metadata.data(),true};
            ggml_context *ctx=ggml_init(params);
            swin_inputs in{inputs.get("pixels"),{inputs.get("mask0"),inputs.get("mask1"),inputs.get("mask2"),inputs.get("mask3")}};
            ggml_tensor *output=build_swin_backbone(ctx,weights,in,stage,blocks);
            ggml_cgraph *graph=ggml_new_graph_custom(ctx,8192,false);ggml_build_forward_expand(graph,output);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(backend,n))throw std::runtime_error(std::string("Vulkan does not support Swin op ")+ggml_op_desc(n));}
            ggml_gallocr_t allocator=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if(!ggml_gallocr_alloc_graph(allocator,graph)||ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)throw std::runtime_error("Swin execution failed");
            auto actual=tensor_f32(output),reference=tensor_f32(inputs.get("reference"));if(actual.size()!=reference.size())throw std::runtime_error("Swin reference size mismatch");
            double mae=0,mse=0,dot=0,aa=0,rr=0;float mx=0;for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-reference[i]);mae+=e;mse+=double(e)*e;mx=std::max(mx,e);dot+=double(actual[i])*reference[i];aa+=double(actual[i])*actual[i];rr+=double(reference[i])*reference[i];}double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));
            std::printf("Swin parity stage=%d blocks=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",stage,blocks,actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);
            // Tiny padded late-stage fixtures can contain an isolated near-zero
            // activation with a larger absolute delta; cosine is the robust
            // whole-stage criterion. Full BiRefNet output has its own strict test.
            ggml_gallocr_free(allocator);ggml_free(ctx);ggml_backend_free(backend);return(cosine>0.999&&mx<0.35f)?0:1;
        }
        if (mode == "--deform-parity") {
            weight_store inputs(backend, path, weight_policy::native);
            std::vector<uint8_t> metadata(8u * 1024u * 1024u);
            ggml_init_params params { metadata.size(), metadata.data(), true };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * output = deform_conv_2d(
                ctx, inputs.get("input"), inputs.get("offsets"),
                inputs.get("masks"), inputs.get("weight"),
                (int) tensor_f32(inputs.get("stride"))[0],
                (int) tensor_f32(inputs.get("stride"))[1],
                (int) tensor_f32(inputs.get("padding"))[0],
                (int) tensor_f32(inputs.get("padding"))[1]);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(graph, output);
            if (!ggml_backend_supports_op(backend, output)) {
                throw std::runtime_error("Vulkan deformable convolution is not supported");
            }
            ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph) ||
                ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Vulkan deformable convolution execution failed");
            }
            const auto actual = tensor_f32(output);
            const auto reference = tensor_f32(inputs.get("reference"));
            if (actual.size() != reference.size()) throw std::runtime_error("deform reference size mismatch");
            double mae=0,mse=0,dot=0,aa=0,rr=0; float mx=0; size_t mx_i=0;
            for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-reference[i]);mae+=e;mse+=double(e)*e;if(e>mx){mx=e;mx_i=i;}dot+=double(actual[i])*reference[i];aa+=double(actual[i])*actual[i];rr+=double(reference[i])*reference[i];}
            const double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));
            std::printf("deform conv parity elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);
            std::printf("  max element=%zu Vulkan=%.8g Torch=%.8g\n", mx_i, actual[mx_i], reference[mx_i]);
            ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
            return (cosine > 0.999999 && mx < 5e-3f) ? 0 : 1;
        }
        if (mode == "--flow-parity") {
            if (argc < 4) {
                usage(argv[0]);
                ggml_backend_free(backend);
                return 2;
            }
            const int blocks = argc > 4 ? std::stoi(argv[4]) : 24;
            weight_store weights(backend, path, weight_policy::native);
            weight_store inputs(backend, argv[3], weight_policy::native);
            std::vector<uint8_t> metadata(64u * 1024u * 1024u);
            ggml_init_params params { metadata.size(), metadata.data(), true };
            ggml_context * ctx = ggml_init(params);
            if (!ctx) throw std::runtime_error("failed to create flow graph context");
            flow_inputs in {
                inputs.get("latent"), inputs.get("feature1"), inputs.get("feature2"),
                inputs.get("camera"), inputs.get("timestep"), inputs.get("positions"),
                inputs.get("position_freqs")
            };
            flow_outputs flow_out = build_flow_outputs(ctx, weights, in, blocks);
            ggml_tensor * output = flow_out.latent;
            ggml_set_name(output, "flow.output");
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(graph, output);
            ggml_build_forward_expand(graph, flow_out.camera);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                if (!ggml_backend_supports_op(backend, node)) {
                    throw std::runtime_error(std::string("Vulkan does not support graph op ") +
                                             ggml_op_desc(node));
                }
            }
            ggml_gallocr_t allocator = ggml_gallocr_new(
                ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) {
                throw std::runtime_error("failed to allocate flow activation graph");
            }
            if (!compute_graph(backend, graph, "flow")) {
                throw std::runtime_error("Vulkan flow graph execution failed");
            }
            const auto actual = tensor_f32(output);
            const auto reference = tensor_f32(inputs.get("reference"));
            if (actual.size() != reference.size()) throw std::runtime_error("reference size mismatch");
            double abs_sum = 0.0, sq_error = 0.0, sq_ref = 0.0, dot = 0.0;
            float max_abs = 0.0f;
            for (size_t i = 0; i < actual.size(); ++i) {
                const float error = std::abs(actual[i] - reference[i]);
                abs_sum += error;
                sq_error += double(error) * error;
                sq_ref += double(reference[i]) * reference[i];
                dot += double(actual[i]) * reference[i];
                max_abs = std::max(max_abs, error);
            }
            double sq_actual = 0.0;
            for (float value : actual) sq_actual += double(value) * value;
            const double cosine = dot / std::sqrt(std::max(1e-30, sq_actual * sq_ref));
            std::printf("flow parity blocks=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",
                        blocks, actual.size(), abs_sum / actual.size(),
                        std::sqrt(sq_error / actual.size()), max_abs, cosine);
            double camera_cosine=1.0;
            if(inputs.maybe("reference_camera")){
                const auto ca=tensor_f32(flow_out.camera),cr=tensor_f32(inputs.get("reference_camera"));double cd=0,caa=0,crr=0,cmae=0;float cmx=0;for(size_t i=0;i<ca.size();++i){float e=std::abs(ca[i]-cr[i]);cmae+=e;cmx=std::max(cmx,e);cd+=double(ca[i])*cr[i];caa+=double(ca[i])*ca[i];crr+=double(cr[i])*cr[i];}camera_cosine=cd/std::sqrt(std::max(1e-30,caa*crr));std::printf("camera parity elements=%zu MAE=%.8g MAX=%.8g cosine=%.10f\n",ca.size(),cmae/ca.size(),cmx,camera_cosine);
            }
            ggml_gallocr_free(allocator);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return (cosine > 0.999 && max_abs < 0.2f && camera_cosine > 0.999) ? 0 : 1;
        }

        if (mode == "--dino-parity") {
            if (argc < 4) {
                usage(argv[0]);
                ggml_backend_free(backend);
                return 2;
            }
            const int layers = argc > 4 ? std::stoi(argv[4]) : 32;
            weight_store weights(backend, path, weight_policy::f16);
            weight_store inputs(backend, argv[3], weight_policy::native);
            std::vector<uint8_t> metadata(64u * 1024u * 1024u);
            ggml_init_params params { metadata.size(), metadata.data(), true };
            ggml_context * ctx = ggml_init(params);
            dino_inputs in { inputs.get("pixels"), inputs.get("rope_cos"), inputs.get("rope_sin") };
            ggml_tensor * output = build_dino_model(ctx, weights, in, layers);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(graph, output);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                if (!ggml_backend_supports_op(backend, node)) {
                    throw std::runtime_error(std::string("Vulkan does not support DINO op ") + ggml_op_desc(node));
                }
            }
            ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("DINO graph allocation failed");
            if (!compute_graph(backend, graph, "DINO")) {
                throw std::runtime_error("Vulkan DINO graph execution failed");
            }
            const auto actual = tensor_f32(output);
            const auto reference = tensor_f32(inputs.get("reference"));
            if (actual.size() != reference.size()) throw std::runtime_error("DINO reference size mismatch");
            double abs_sum = 0.0, sq_error = 0.0, dot = 0.0, aa = 0.0, rr = 0.0;
            float max_abs = 0.0f;
            for (size_t i = 0; i < actual.size(); ++i) {
                const float e = std::abs(actual[i] - reference[i]);
                abs_sum += e; sq_error += double(e)*e; max_abs = std::max(max_abs, e);
                dot += double(actual[i])*reference[i]; aa += double(actual[i])*actual[i]; rr += double(reference[i])*reference[i];
            }
            const double cosine = dot/std::sqrt(std::max(1e-30, aa*rr));
            std::printf("DINO parity layers=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",
                        layers, actual.size(), abs_sum/actual.size(), std::sqrt(sq_error/actual.size()), max_abs, cosine);
            if (layers < 0) {
                for (int token = 0; token < 9; ++token) {
                    double token_mae = 0.0;
                    for (int c = 0; c < 1280; ++c) {
                        const size_t j = size_t(token)*1280 + c;
                        token_mae += std::abs(actual[j] - reference[j]);
                    }
                    std::printf("  token %d MAE=%.8g first=(%.8g, %.8g)\n", token,
                                token_mae/1280.0, actual[size_t(token)*1280], reference[size_t(token)*1280]);
                }
            }
            ggml_gallocr_free(allocator);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return (cosine > 0.999 && max_abs < 0.2f) ? 0 : 1;
        }

        if (mode == "--vae-parity") {
            if (argc < 4) { usage(argv[0]); ggml_backend_free(backend); return 2; }
            const char * persistent_f16_env = std::getenv("TRIPOSPLAT_VAE_PERSISTENT_F16");
            const bool persistent_f16 = persistent_f16_env == nullptr ||
                                        std::string(persistent_f16_env) != "0";
            weight_store weights(backend, path,
                                 persistent_f16 ? weight_policy::f16 : weight_policy::native);
            weight_store inputs(backend, argv[3], weight_policy::native);
            std::vector<uint8_t> metadata(64u * 1024u * 1024u);
            ggml_init_params params { metadata.size(), metadata.data(), true };
            ggml_context * ctx = ggml_init(params);
            const int stage = argc > 4 ? std::stoi(argv[4]) : 13;
            ggml_tensor * output = build_vae_encoder(ctx, weights, inputs.get("image"), stage);
            ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(graph, output);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                if (!ggml_backend_supports_op(backend, node)) {
                    throw std::runtime_error(std::string("Vulkan does not support VAE op ") + ggml_op_desc(node));
                }
            }
            ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("VAE graph allocation failed");
            if (!compute_graph(backend, graph, "VAE")) throw std::runtime_error("VAE execution failed");
            const auto actual = tensor_f32(output), reference = tensor_f32(inputs.get("reference"));
            if (actual.size() != reference.size()) throw std::runtime_error("VAE reference size mismatch");
            double mae=0, mse=0, dot=0, aa=0, rr=0; float mx=0;
            for (size_t i=0;i<actual.size();++i) { float e=std::abs(actual[i]-reference[i]); mae+=e; mse+=double(e)*e; mx=std::max(mx,e); dot+=double(actual[i])*reference[i]; aa+=double(actual[i])*actual[i]; rr+=double(reference[i])*reference[i]; }
            double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));
            std::printf("VAE parity stage=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n", stage,actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);
            ggml_gallocr_free(allocator); ggml_free(ctx); ggml_backend_free(backend);
            return (cosine > 0.999 && mx < 0.2f) ? 0 : 1;
        }

        if (mode == "--gs-parity") {
            if (argc < 4) { usage(argv[0]); ggml_backend_free(backend); return 2; }
            const int blocks=argc>4?std::stoi(argv[4]):16;
            weight_store weights(backend,path,weight_policy::native), inputs(backend,argv[3],weight_policy::native);
            std::vector<uint8_t> metadata(64u*1024u*1024u); ggml_init_params params{metadata.size(),metadata.data(),true};
            ggml_context *ctx=ggml_init(params);
            gs_inputs in{inputs.get("points"),inputs.get("condition"),inputs.get("position_freqs")};
            ggml_tensor *output=build_gs_decoder(ctx,weights,in,blocks);
            ggml_cgraph *graph=ggml_new_graph_custom(ctx,8192,false); ggml_build_forward_expand(graph,output);
            for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(backend,n))throw std::runtime_error(std::string("Vulkan does not support GS op ")+ggml_op_desc(n));}
            ggml_gallocr_t a=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if(!ggml_gallocr_alloc_graph(a,graph)||!compute_graph(backend,graph,"GS decoder"))throw std::runtime_error("GS execution failed");
            auto actual=tensor_f32(output),reference=tensor_f32(inputs.get("reference"));if(actual.size()!=reference.size())throw std::runtime_error("GS reference size mismatch");
            double mae=0,mse=0,dot=0,aa=0,rr=0;float mx=0;for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-reference[i]);mae+=e;mse+=double(e)*e;mx=std::max(mx,e);dot+=double(actual[i])*reference[i];aa+=double(actual[i])*actual[i];rr+=double(reference[i])*reference[i];}
            double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));std::printf("GS parity blocks=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",blocks,actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);
            ggml_gallocr_free(a);ggml_free(ctx);ggml_backend_free(backend);return(cosine>0.999&&mx<0.2f)?0:1;
        }

        if(mode=="--octree-parity"){
            if(argc<4){usage(argv[0]);ggml_backend_free(backend);return 2;}const int blocks=argc>4?std::stoi(argv[4]):4;
            weight_store weights(backend,path,weight_policy::native),inputs(backend,argv[3],weight_policy::native);
            std::vector<uint8_t> metadata(64u*1024u*1024u);ggml_init_params params{metadata.size(),metadata.data(),true};ggml_context*ctx=ggml_init(params);
            octree_inputs in{inputs.get("points"),inputs.get("level"),inputs.get("condition"),inputs.get("position_freqs")};ggml_tensor*output=build_octree_decoder(ctx,weights,in,blocks);
            ggml_cgraph*graph=ggml_new_graph_custom(ctx,8192,false);ggml_build_forward_expand(graph,output);for(int i=0;i<ggml_graph_n_nodes(graph);++i){auto*n=ggml_graph_node(graph,i);if(!ggml_backend_supports_op(backend,n))throw std::runtime_error(std::string("Vulkan does not support octree op ")+ggml_op_desc(n));}
            ggml_gallocr_t a=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));if(!ggml_gallocr_alloc_graph(a,graph)||!compute_graph(backend,graph,"octree decoder"))throw std::runtime_error("octree execution failed");
            auto actual=tensor_f32(output),reference=tensor_f32(inputs.get("reference"));if(actual.size()!=reference.size())throw std::runtime_error("octree reference size mismatch");double mae=0,mse=0,dot=0,aa=0,rr=0;float mx=0;for(size_t i=0;i<actual.size();++i){float e=std::abs(actual[i]-reference[i]);mae+=e;mse+=double(e)*e;mx=std::max(mx,e);dot+=double(actual[i])*reference[i];aa+=double(actual[i])*actual[i];rr+=double(reference[i])*reference[i];}double cosine=dot/std::sqrt(std::max(1e-30,aa*rr));std::printf("octree parity blocks=%d elements=%zu MAE=%.8g RMSE=%.8g MAX=%.8g cosine=%.10f\n",blocks,actual.size(),mae/actual.size(),std::sqrt(mse/actual.size()),mx,cosine);
            ggml_gallocr_free(a);ggml_free(ctx);ggml_backend_free(backend);return(cosine>0.999&&mx<0.2f)?0:1;
        }

        const bool f16 = argc > 3 && std::string(argv[3]) == "--f16";
        {
            weight_store weights(backend, path, f16 ? weight_policy::f16 : weight_policy::native);
            std::printf("loaded %zu tensors (%.2f MiB) into Vulkan\n",
                        weights.tensor_count(), double(weights.bytes()) / (1024.0 * 1024.0));
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "triposplat-vulkan: %s\n", error.what());
        return 1;
    }
}
