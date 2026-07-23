#include "model-store.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace triposplat {
namespace {

namespace fs = std::filesystem;

constexpr std::array<const char *, 5> kFiles {{
    "background_removal/birefnet.safetensors",
    "clip_vision/dino_v3_vit_h.safetensors",
    "vae/flux2-vae.safetensors",
    "diffusion_models/triposplat_fp16.safetensors",
    "vae/triposplat_vae_decoder_fp16.safetensors",
}};

std::string shell_quote(const std::string & value) {
#ifdef _WIN32
    std::string result = "\"";
    for (char c : value) result += c == '"' ? "\\\"" : std::string(1, c);
    return result + "\"";
#else
    std::string result = "'";
    for (char c : value) result += c == '\'' ? "'\\''" : std::string(1, c);
    return result + "'";
#endif
}

void download_one(const model_store_config & config, const std::string & relative) {
    const fs::path destination = fs::path(config.directory) / relative;
    if (fs::is_regular_file(destination) && fs::file_size(destination) > 0) return;
    fs::create_directories(destination.parent_path());
    const fs::path partial = destination.string() + ".part";
    const std::string url = "https://huggingface.co/" + config.repository +
                            "/resolve/" + config.revision + "/" + relative;
#ifdef _WIN32
    const std::string command = "curl.exe -fL --retry 3 -C - -o " +
        shell_quote(partial.string()) + " " + shell_quote(url);
#else
    const std::string command = "curl -fL --retry 3 -C - -o " +
        shell_quote(partial.string()) + " " + shell_quote(url);
#endif
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("Hugging Face download failed: " + relative);
    }
    if (!fs::is_regular_file(partial) || fs::file_size(partial) == 0) {
        throw std::runtime_error("Hugging Face returned an empty model: " + relative);
    }
    fs::rename(partial, destination);
}

} // namespace

void download_models(const model_store_config & config) {
    if (config.directory.empty()) throw std::invalid_argument("model directory is empty");
    for (const char * file : kFiles) download_one(config, file);
}

model_paths resolve_models(const model_store_config & config) {
    if (config.allow_download) download_models(config);
    auto require = [&](const char * relative) {
        const fs::path path = fs::path(config.directory) / relative;
        if (!fs::is_regular_file(path) || fs::file_size(path) == 0) {
            throw std::runtime_error("missing model " + path.string() +
                                     " (use `download` or `--download`)");
        }
        return fs::absolute(path).string();
    };
    return {require(kFiles[0]), require(kFiles[1]), require(kFiles[2]),
            require(kFiles[3]), require(kFiles[4])};
}

} // namespace triposplat
