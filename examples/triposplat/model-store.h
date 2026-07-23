#pragma once

#include <string>

namespace triposplat {

struct model_store_config {
    std::string directory;
    std::string repository = "VAST-AI/TripoSplat";
    std::string revision = "main";
    bool allow_download = false;
};

struct model_paths {
    std::string biref;
    std::string dino;
    std::string vae_encoder;
    std::string flow;
    std::string decoder;
};

model_paths resolve_models(const model_store_config & config);
void download_models(const model_store_config & config);

} // namespace triposplat
