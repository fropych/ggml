#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)
build_root="${repo_root}/builds/windows-x86_64"
deps_root="${repo_root}/builds/windows-deps"
deps_prefix="${deps_root}/prefix"
toolchain="${repo_root}/cmake/toolchains/x86_64-w64-mingw32.cmake"
webp_version=1.3.2
webp_sha256=2a499607df669e40258e53d0ade8035ba4ec0175244869d1025d460562aa09b4
webp_archive="${deps_root}/libwebp-${webp_version}.tar.gz"
webp_source="${deps_root}/src/libwebp-${webp_version}"
webp_build="${deps_root}/libwebp-build"

vulkan_sdk="${VULKAN_SDK:-}"
local_sdk="${repo_root}/../.tools/vulkan-sdk-1.4.350.1/x86_64"
if [[ -z "${vulkan_sdk}" && -x "${local_sdk}/bin/glslc" ]]; then
    vulkan_sdk="${local_sdk}"
fi
if [[ -z "${vulkan_sdk}" || ! -x "${vulkan_sdk}/bin/glslc" ]]; then
    echo "Set VULKAN_SDK to a Linux Vulkan SDK containing bin/glslc and include/vulkan." >&2
    exit 1
fi

for command in cmake ninja curl tar sha256sum \
        x86_64-w64-mingw32-gcc x86_64-w64-mingw32-g++ \
        x86_64-w64-mingw32-dlltool; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Missing build command: ${command}" >&2
        exit 1
    fi
done
if [[ ! -f /usr/include/nlohmann/json.hpp ]]; then
    echo "Missing /usr/include/nlohmann/json.hpp (install nlohmann-json3-dev)." >&2
    exit 1
fi

mkdir -p "${deps_root}/src" "${deps_prefix}/include/nlohmann" "${deps_prefix}/lib"

if [[ ! -f "${webp_archive}" ]]; then
    curl -fL --retry 3 \
        -o "${webp_archive}" \
        "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-${webp_version}.tar.gz"
fi
printf '%s  %s\n' "${webp_sha256}" "${webp_archive}" | sha256sum --check -

if [[ ! -f "${webp_source}/CMakeLists.txt" ]]; then
    tar -xzf "${webp_archive}" -C "${deps_root}/src"
fi
cmake -E copy_directory /usr/include/nlohmann "${deps_prefix}/include/nlohmann"

x86_64-w64-mingw32-dlltool \
    -d "${script_dir}/cmake/vulkan-1.def" \
    -l "${deps_prefix}/lib/libvulkan-1.dll.a" \
    -D vulkan-1.dll

cmake -S "${webp_source}" -B "${webp_build}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${deps_prefix}" \
    -DBUILD_SHARED_LIBS=OFF \
    -DWEBP_BUILD_ANIM_UTILS=OFF \
    -DWEBP_BUILD_CWEBP=OFF \
    -DWEBP_BUILD_DWEBP=OFF \
    -DWEBP_BUILD_GIF2WEBP=OFF \
    -DWEBP_BUILD_IMG2WEBP=OFF \
    -DWEBP_BUILD_VWEBP=OFF \
    -DWEBP_BUILD_WEBPINFO=OFF \
    -DWEBP_BUILD_WEBPMUX=OFF \
    -DWEBP_BUILD_EXTRAS=OFF
cmake --build "${webp_build}" --target install --parallel

cmake -S "${repo_root}" -B "${build_root}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_VULKAN=ON \
    -DGGML_CPU=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BUILD_EXAMPLES=ON \
    -DGGML_BUILD_TESTS=OFF \
    -DGGML_BUILD_TOOLS=OFF \
    -DVulkan_INCLUDE_DIR="${vulkan_sdk}/include" \
    -DVulkan_LIBRARY="${deps_prefix}/lib/libvulkan-1.dll.a" \
    -DVulkan_GLSLC_EXECUTABLE="${vulkan_sdk}/bin/glslc" \
    -DSPIRV-Headers_DIR="${vulkan_sdk}/share/cmake/SPIRV-Headers" \
    -DTRIPOSPLAT_WEBP_STATIC_LIBRARY="${deps_prefix}/lib/libwebp.a" \
    -DTRIPOSPLAT_SHARPYUV_STATIC_LIBRARY="${deps_prefix}/lib/libsharpyuv.a" \
    -DTRIPOSPLAT_WEBP_INCLUDE_DIR="${deps_prefix}/include" \
    -DTRIPOSPLAT_NLOHMANN_JSON_INCLUDE_DIR="${deps_prefix}/include"
cmake --build "${build_root}" --target triposplat-package --parallel

echo "Created ${build_root}/triposplat-windows-x86_64.zip"
