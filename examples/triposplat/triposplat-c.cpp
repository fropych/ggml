#include "triposplat-c.h"
#include "pipeline.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

struct ts_pipeline {
    triposplat::pipeline value;
    explicit ts_pipeline(int device) : value(device) {}
};

namespace {
void set_error(char * output, size_t size, const std::string & message) {
    if (!output || size == 0) return;
    const size_t count = std::min(size - 1, message.size());
    std::memcpy(output, message.data(), count);
    output[count] = '\0';
}
}

extern "C" ts_pipeline * ts_create(int vulkan_device, char * error, size_t error_size) {
    try {
        return new ts_pipeline(vulkan_device);
    } catch (const std::exception & exception) {
        set_error(error, error_size, exception.what());
        return nullptr;
    }
}

extern "C" int ts_generate(ts_pipeline * instance, const ts_generate_options * input,
                            char * error, size_t error_size) {
    try {
        if (!instance || !input) throw std::invalid_argument("null ts_generate argument");
        triposplat::generate_options options;
        options.input_image = input->input_image ? input->input_image : "";
        options.output_prefix = input->output_prefix ? input->output_prefix : "triposplat";
        options.asset_directory = input->asset_directory ? input->asset_directory : "";
        options.models.directory = input->model_directory ? input->model_directory : "";
        options.models.repository = input->huggingface_repository ?
            input->huggingface_repository : "VAST-AI/TripoSplat";
        options.models.revision = input->revision ? input->revision : "main";
        options.models.allow_download = input->allow_download != 0;
        options.seed = input->seed;
        options.steps = input->steps;
        options.guidance = input->guidance;
        options.num_gaussians = input->num_gaussians;
        options.erode_radius = input->erode_radius;
        instance->value.generate(options);
        return 0;
    } catch (const std::exception & exception) {
        set_error(error, error_size, exception.what());
        return -1;
    }
}

extern "C" const char * ts_device_description(const ts_pipeline * instance) {
    return instance ? instance->value.device_description().c_str() : "";
}

extern "C" void ts_destroy(ts_pipeline * instance) {
    delete instance;
}
