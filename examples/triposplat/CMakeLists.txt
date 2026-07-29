find_library(TRIPOSPLAT_WEBP_STATIC_LIBRARY
    NAMES libwebp.a webp.a webp REQUIRED)
find_library(TRIPOSPLAT_SHARPYUV_STATIC_LIBRARY
    NAMES libsharpyuv.a sharpyuv.a sharpyuv REQUIRED)
find_path(TRIPOSPLAT_WEBP_INCLUDE_DIR webp/decode.h REQUIRED)
find_path(TRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)
find_package(Vulkan COMPONENTS glslc REQUIRED)
find_package(Threads REQUIRED)

set(TRIPOSPLAT_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(TRIPOSPLAT_GENERATED_SHADER_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated-shaders")
file(MAKE_DIRECTORY "${TRIPOSPLAT_GENERATED_SHADER_DIR}")
set(TRIPOSPLAT_SHADER_HEADERS)
foreach(SHADER_NAME voxel_preprocess voxel_integrate voxel_finalize)
    set(SHADER_SOURCE "${TRIPOSPLAT_SHADER_DIR}/${SHADER_NAME}.comp")
    set(SHADER_SPV
        "${TRIPOSPLAT_GENERATED_SHADER_DIR}/${SHADER_NAME}.spv")
    set(SHADER_HEADER
        "${TRIPOSPLAT_GENERATED_SHADER_DIR}/${SHADER_NAME}.spv.h")
    add_custom_command(
        OUTPUT "${SHADER_SPV}"
        COMMAND "${Vulkan_GLSLC_EXECUTABLE}" -O
                --target-env=vulkan1.2
                -o "${SHADER_SPV}" "${SHADER_SOURCE}"
        DEPENDS "${SHADER_SOURCE}"
        VERBATIM)
    add_custom_command(
        OUTPUT "${SHADER_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SHADER_SPV}"
                "-DOUTPUT=${SHADER_HEADER}"
                "-DSYMBOL=triposplat_${SHADER_NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-spirv.cmake"
        DEPENDS "${SHADER_SPV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-spirv.cmake"
        VERBATIM)
    list(APPEND TRIPOSPLAT_SHADER_HEADERS "${SHADER_HEADER}")
endforeach()

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
    voxel-converter.cpp
    ${TRIPOSPLAT_SHADER_HEADERS}
)
set_target_properties(triposplat-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(triposplat-core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${TRIPOSPLAT_WEBP_INCLUDE_DIR}
        ${TRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR}
        ${TRIPOSPLAT_GENERATED_SHADER_DIR})
target_link_libraries(triposplat-core PUBLIC
    ggml
    ggml-vulkan
    Vulkan::Vulkan
    ${TRIPOSPLAT_WEBP_STATIC_LIBRARY}
    ${TRIPOSPLAT_SHARPYUV_STATIC_LIBRARY}
)
target_compile_features(triposplat-core PUBLIC cxx_std_17)

add_library(triposplat-civetweb STATIC
    third_party/civetweb/src/civetweb.c)
target_include_directories(triposplat-civetweb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/civetweb/include)
target_compile_definitions(triposplat-civetweb PRIVATE
    NO_SSL
    NO_CGI
    NO_CACHING)
target_link_libraries(triposplat-civetweb PUBLIC Threads::Threads)
if(WIN32)
    target_link_libraries(triposplat-civetweb PUBLIC ws2_32)
endif()

add_library(triposplat-artifact-store STATIC
    artifact-store.cpp)
target_include_directories(triposplat-artifact-store
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR})
target_compile_features(triposplat-artifact-store PUBLIC cxx_std_17)

add_library(triposplat-server STATIC
    job-manager.cpp
    server.cpp)
target_include_directories(triposplat-server
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${TRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(triposplat-server PUBLIC
    triposplat-artifact-store
    triposplat-core
    triposplat-civetweb)
target_compile_features(triposplat-server PUBLIC cxx_std_17)

set(TARGET triposplat-vulkan)
add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE triposplat-server)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
add_custom_command(TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${TARGET}>/assets"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/assets/flow_positions.safetensors"
            "$<TARGET_FILE_DIR:${TARGET}>/assets/flow_positions.safetensors"
    VERBATIM)

if(TRIPOSPLAT_BUILD_TESTS)
    add_executable(triposplat-test-artifact-store
        tests/test_artifact_store.cpp)
    target_link_libraries(triposplat-test-artifact-store PRIVATE
        triposplat-artifact-store)
    target_compile_features(
        triposplat-test-artifact-store PRIVATE cxx_std_17)
    add_test(
        NAME triposplat-artifact-store
        COMMAND triposplat-test-artifact-store)

    add_executable(triposplat-test-http-api
        tests/test_http_api.cpp)
    target_include_directories(triposplat-test-http-api PRIVATE
        ${TRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(triposplat-test-http-api PRIVATE
        triposplat-server)
    target_compile_features(
        triposplat-test-http-api PRIVATE cxx_std_17)
    add_dependencies(triposplat-test-http-api triposplat-vulkan)

    set(TRIPOSPLAT_HTTP_CONTRACT_INPUT "" CACHE FILEPATH
        "Input image for the TripoSplat HTTP GPU contract")
    set(TRIPOSPLAT_HTTP_CONTRACT_MODEL_DIR "" CACHE PATH
        "Checkpoint directory for the TripoSplat HTTP GPU contract")
    set(TRIPOSPLAT_HTTP_CONTRACT_DEVICE "0" CACHE STRING
        "Vulkan device for the TripoSplat HTTP GPU contract")
    if(TRIPOSPLAT_HTTP_CONTRACT_INPUT AND
       TRIPOSPLAT_HTTP_CONTRACT_MODEL_DIR)
        add_test(
            NAME triposplat-http-api
            COMMAND triposplat-test-http-api
                --binary $<TARGET_FILE:triposplat-vulkan>
                --input ${TRIPOSPLAT_HTTP_CONTRACT_INPUT}
                --model-dir ${TRIPOSPLAT_HTTP_CONTRACT_MODEL_DIR}
                --device ${TRIPOSPLAT_HTTP_CONTRACT_DEVICE})
        set_tests_properties(triposplat-http-api PROPERTIES
            LABELS "contract;gpu"
            RESOURCE_LOCK triposplat-gpu
            TIMEOUT 1800)
    endif()
endif()

if(WIN32)
    if(MINGW)
        target_link_options(${TARGET} PRIVATE
            -static -static-libgcc -static-libstdc++)
    endif()
    set(TRIPOSPLAT_PACKAGE_NAME "triposplat-windows-x86_64")
    set(TRIPOSPLAT_PACKAGE_EXTENSION ".zip")
else()
    set_target_properties(${TARGET} PROPERTIES
        BUILD_RPATH "$ORIGIN;$ORIGIN/../src;$ORIGIN/../src/ggml-vulkan"
        BUILD_RPATH_USE_ORIGIN TRUE
        INSTALL_RPATH "$ORIGIN"
    )
    set(TRIPOSPLAT_PACKAGE_NAME "triposplat-linux-${CMAKE_SYSTEM_PROCESSOR}")
    set(TRIPOSPLAT_PACKAGE_EXTENSION ".tar.gz")
endif()

set(TRIPOSPLAT_PACKAGE_ROOT "${CMAKE_BINARY_DIR}/package")
set(TRIPOSPLAT_PACKAGE_DIR "${TRIPOSPLAT_PACKAGE_ROOT}/${TRIPOSPLAT_PACKAGE_NAME}")
set(TRIPOSPLAT_PACKAGE_ARCHIVE
    "${CMAKE_BINARY_DIR}/${TRIPOSPLAT_PACKAGE_NAME}${TRIPOSPLAT_PACKAGE_EXTENSION}")
file(MAKE_DIRECTORY "${TRIPOSPLAT_PACKAGE_ROOT}")

install(TARGETS ${TARGET}
    RUNTIME DESTINATION .
    COMPONENT triposplat-portable)
if(NOT WIN32)
    install(FILES "$<TARGET_FILE:ggml-vulkan>"
        DESTINATION .
        RENAME "libggml-vulkan.so.0"
        COMPONENT triposplat-portable)
    install(FILES "$<TARGET_FILE:ggml>"
        DESTINATION .
        RENAME "libggml.so.0"
        COMPONENT triposplat-portable)
    install(FILES "$<TARGET_FILE:ggml-base>"
        DESTINATION .
        RENAME "libggml-base.so.0"
        COMPONENT triposplat-portable)
endif()
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/assets/flow_positions.safetensors"
    DESTINATION assets
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/VOXEL_FORMAT.md"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/VOXEL_CHUNKING_BENCHMARK.md"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/VOXEL_SURFACE_BENCHMARK.md"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES "${CMAKE_SOURCE_DIR}/LICENSE"
    DESTINATION .
    COMPONENT triposplat-portable)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/civetweb/LICENSE.md"
    DESTINATION licenses
    RENAME CivetWeb-LICENSE.md
    COMPONENT triposplat-portable)

if(WIN32)
    add_custom_target(triposplat-package
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${TRIPOSPLAT_PACKAGE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E rm -f "${TRIPOSPLAT_PACKAGE_ARCHIVE}"
        COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}"
                --prefix "${TRIPOSPLAT_PACKAGE_DIR}"
                --component triposplat-portable
        COMMAND ${CMAKE_COMMAND} -E tar cf "${TRIPOSPLAT_PACKAGE_ARCHIVE}"
                --format=zip "${TRIPOSPLAT_PACKAGE_NAME}"
        WORKING_DIRECTORY "${TRIPOSPLAT_PACKAGE_ROOT}"
        DEPENDS ${TARGET}
        COMMENT "Creating portable ${TRIPOSPLAT_PACKAGE_NAME}.zip"
    )
else()
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
endif()
