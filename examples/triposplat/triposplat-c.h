#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(TRIPOSPLAT_SHARED_BUILD)
#    define TS_API __declspec(dllexport)
#  else
#    define TS_API
#  endif
#else
#  define TS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ts_pipeline ts_pipeline;

typedef struct ts_generate_options {
    const char * input_image;
    const char * output_prefix;
    const char * model_directory;
    const char * asset_directory;
    const char * huggingface_repository;
    const char * revision;
    uint64_t seed;
    int steps;
    float guidance;
    size_t num_gaussians;
    int erode_radius;
    int allow_download;
} ts_generate_options;

TS_API ts_pipeline * ts_create(int vulkan_device, char * error, size_t error_size);
TS_API int ts_generate(ts_pipeline * instance, const ts_generate_options * options,
                       char * error, size_t error_size);
TS_API const char * ts_device_description(const ts_pipeline * instance);
TS_API void ts_destroy(ts_pipeline * instance);

#ifdef __cplusplus
}
#endif
