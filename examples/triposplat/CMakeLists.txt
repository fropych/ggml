find_library(TRIPOSPLAT_WEBP_STATIC_LIBRARY NAMES libwebp.a REQUIRED)
find_library(TRIPOSPLAT_SHARPYUV_STATIC_LIBRARY NAMES libsharpyuv.a REQUIRED)

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
target_link_libraries(triposplat-core PUBLIC
    ggml
    ggml-vulkan
    ${TRIPOSPLAT_WEBP_STATIC_LIBRARY}
    ${TRIPOSPLAT_SHARPYUV_STATIC_LIBRARY}
)
target_compile_features(triposplat-core PUBLIC cxx_std_17)

set(TARGET triposplat-vulkan)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE triposplat-core)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
set_target_properties(${TARGET} PROPERTIES
    BUILD_RPATH "$ORIGIN;$ORIGIN/../src;$ORIGIN/../src/ggml-vulkan"
    BUILD_RPATH_USE_ORIGIN TRUE
    INSTALL_RPATH "$ORIGIN"
)

set(TRIPOSPLAT_PACKAGE_NAME "triposplat-linux-${CMAKE_SYSTEM_PROCESSOR}")
set(TRIPOSPLAT_PACKAGE_ROOT "${CMAKE_BINARY_DIR}/package")
set(TRIPOSPLAT_PACKAGE_DIR "${TRIPOSPLAT_PACKAGE_ROOT}/${TRIPOSPLAT_PACKAGE_NAME}")
set(TRIPOSPLAT_PACKAGE_ARCHIVE
    "${CMAKE_BINARY_DIR}/${TRIPOSPLAT_PACKAGE_NAME}.tar.gz")
file(MAKE_DIRECTORY "${TRIPOSPLAT_PACKAGE_ROOT}")

install(TARGETS ${TARGET}
    RUNTIME DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "$<TARGET_FILE:ggml-vulkan>"
    DESTINATION .
    RENAME "libggml-vulkan.so.0"
    COMPONENT triposplat-portable)
install(FILES "$<TARGET_FILE:ggml-base>"
    DESTINATION .
    RENAME "libggml-base.so.0"
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/flow_positions.safetensors"
    DESTINATION assets
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION .
    COMPONENT triposplat-portable)

add_custom_target(triposplat-package
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${TRIPOSPLAT_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E rm -f "${TRIPOSPLAT_PACKAGE_ARCHIVE}"
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
            --prefix "${TRIPOSPLAT_PACKAGE_DIR}"
            --component triposplat-portable
    COMMAND ${CMAKE_COMMAND} -E tar czf "${TRIPOSPLAT_PACKAGE_ARCHIVE}"
            --format=gnutar "${TRIPOSPLAT_PACKAGE_NAME}"
    WORKING_DIRECTORY "${TRIPOSPLAT_PACKAGE_ROOT}"
    DEPENDS ${TARGET}
    COMMENT "Creating portable ${TRIPOSPLAT_PACKAGE_NAME}.tar.gz"
)
