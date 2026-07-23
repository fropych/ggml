find_library(TRIPOSPLAT_WEBP_LIBRARY webp REQUIRED)

add_library(triposplat-core STATIC
    safetensors.cpp
    host-tensor-io.cpp
    triposplat-ops.cpp
    flow-model.cpp
    dino-model.cpp
    vae-model.cpp
    decoder-model.cpp
    decoder-runtime.cpp
    e2e-worker.cpp
    biref-model.cpp
    image-io.cpp
    model-store.cpp
    pipeline.cpp
    triposplat-c.cpp
)
set_target_properties(triposplat-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(triposplat-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(triposplat-core PUBLIC ggml ggml-vulkan ${TRIPOSPLAT_WEBP_LIBRARY})
target_compile_features(triposplat-core PUBLIC cxx_std_17)
target_compile_definitions(triposplat-core PRIVATE
    TRIPOSPLAT_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets")

set(TARGET triposplat-vulkan)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE triposplat-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
