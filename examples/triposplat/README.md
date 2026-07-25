# TripoSplat on ggml/Vulkan

This directory is a Vulkan-only port of the neural networks used by the
neighbouring `TripoSplat` PyTorch project. The executable creates one ggml
Vulkan backend and deliberately does not create a ggml CPU backend.

Implemented networks:

- DINOv3 ViT-H/16 image encoder (32 transformer blocks);
- Flux2 VAE encoder;
- TripoSplat flow model (2 noise refiners, 2 context refiners, 24 joint
  blocks, latent and camera heads, reusable CFG Euler graph);
- octree probability decoder (4 blocks);
- elastic Gaussian decoder (16 blocks);
- BiRefNet with the full Swin-L backbone and decoder.

The Vulkan backend was extended with the operations absent from upstream ggml:
deformable convolution (bilinear im2col followed by Vulkan matrix
multiplication), image-to-patches, window partition and window unpartition.
Weights are read directly from the original safetensors checkpoints; no model
conversion step is required.

## Build

The build may omit the CPU backend completely:

```sh
cmake -S . -B builds/vulkan-cm2 -G Ninja \
  -DGGML_VULKAN=ON -DGGML_CPU=OFF -DGGML_BUILD_EXAMPLES=ON
cmake --build builds/vulkan-cm2 --target triposplat-vulkan -j
```

Run on a selected Vulkan device:

```sh
GGML_VK_VISIBLE_DEVICES=0 \
  ./builds/vulkan-cm2/bin/triposplat-vulkan \
  --load ../TripoSplat/ckpts/diffusion_models/triposplat_fp16.safetensors
```

`--load ... --f16` converts F32/BF16 checkpoint tensors to F16 while uploading
them. I64 lookup tables are converted to I32 because that is the Vulkan-native
index representation used by this port.

## Portable Linux x86-64 bundle

Build a relocatable archive after configuring the Vulkan build:

```sh
cmake --build builds/vulkan-cm2 --target triposplat-package -j
```

The result is:

```text
builds/vulkan-cm2/triposplat-linux-x86_64.tar.gz
```

Its layout is:

```text
triposplat-linux-x86_64/
├── triposplat-vulkan
├── libggml-vulkan.so.0
├── libggml-base.so.0
├── assets/flow_positions.safetensors
├── README.md
└── LICENSE
```

The executable has `RUNPATH=$ORIGIN`, so it finds the bundled ggml libraries
next to itself without `LD_LIBRARY_PATH`. WebP and SharpYUV are linked
statically. The target machine still supplies the normal Linux runtime,
`libgomp.so.1`, `libvulkan.so.1`, and an NVIDIA Vulkan driver. The package is
intended for another x86-64 machine running the same Linux distribution (or a
distribution with a compatible glibc/libstdc++).

```sh
tar -xzf triposplat-linux-x86_64.tar.gz
cd triposplat-linux-x86_64
./triposplat-vulkan generate input.webp \
  --model-dir ./ckpts --output output
```

The CLI resolves `assets/` relative to its own executable path, so it may be
started from any working directory.

## Cross-build Windows x86-64 on Linux

Install CMake, Ninja, curl, `nlohmann-json3-dev`, and the MinGW-w64 x86-64 C/C++
toolchain. Point `VULKAN_SDK` at a Linux Vulkan SDK; its native `glslc` compiles
the embedded SPIR-V while MinGW builds the Windows host code:

```sh
VULKAN_SDK=/opt/vulkan-sdk/x86_64 \
  ./examples/triposplat/build-windows-cross.sh
```

The script downloads and verifies libwebp 1.3.2, cross-builds WebP and
SharpYUV statically, creates the Vulkan loader import library, and writes:

```text
builds/windows-x86_64/triposplat-windows-x86_64.zip
```

The archive contains the console executable, runtime asset, README, and
license. ggml, the Vulkan backend, WebP, SharpYUV, libstdc++, and libgcc are
linked statically; OpenMP and the CPU backend are disabled. The executable
therefore imports only `KERNEL32.dll`, `msvcrt.dll`, and `vulkan-1.dll`.
The first two are Windows system libraries. `vulkan-1.dll` is supplied by the
installed NVIDIA/AMD/Intel graphics driver; do not copy a Linux Vulkan library
into the package.

Run from PowerShell or `cmd.exe`:

```powershell
.\triposplat-vulkan.exe download --model-dir .\ckpts
.\triposplat-vulkan.exe generate .\input.webp `
  --model-dir .\ckpts --output .\output
```

The downloader invokes the `curl.exe` bundled with current Windows versions.
For older Windows installations, place a compatible `curl.exe` on `PATH`.

## Native C++ end-to-end CLI

`triposplat-vulkan generate` runs image decoding/preprocessing, DINO, stochastic
VAE encoding, flow sampling, octree selection, Gaussian decoding, and PLY/SPLAT
export from C++. Python and Torch are not used. All learned layers execute on
the selected Vulkan device; CPU code is limited to image/control/RNG/export
work.

```sh
GGML_VK_VISIBLE_DEVICES=0 \
  ./builds/vulkan-cm2/bin/triposplat-vulkan generate \
  ../TripoSplat/static/example_inputs/building_stone_house.webp \
  --model-dir ../TripoSplat/ckpts \
  --output ../TripoSplat/e2e_outputs/building_stone_house_cpp \
  --steps 20 --guidance 3.0 --num-gaussians 32768 --seed 42
```

The CLI accepts WebP, PNG, and JPEG. It can fetch missing weights directly from
Hugging Face using the system `curl` executable:

```sh
./builds/vulkan-cm2/bin/triposplat-vulkan download \
  --model-dir ./ckpts --repo VAST-AI/TripoSplat

./builds/vulkan-cm2/bin/triposplat-vulkan generate input.webp \
  --model-dir ./ckpts --download --output output
```

Downloads use resumable `.part` files and are renamed only after a successful
transfer. `--revision` pins a Hugging Face branch, tag, or commit.

The implementation is split into the reusable `triposplat-core` library and a
thin CLI. `pipeline.h` is the C++ API and `triposplat-c.h` is the stable C ABI
intended for a future JNI/Windows DLL wrapper. A pipeline owns one Vulkan
backend and can serve multiple `generate()` calls.

The current core invokes each verified model stage in-process and exchanges
stage results through temporary safetensors. The public API does not expose
those files, so they can later be replaced with persistent GPU buffers without
changing CLI, C ABI, or JNI callers.

## Python reference orchestrator

The host orchestrator depends only on NumPy, Pillow and safetensors. It does not
import Torch. All learned stages (BiRefNet when needed, DINO, Flux2 VAE, Flow,
octree and Gaussian decoder) execute through the Vulkan-only binary:

```sh
GGML_VK_VISIBLE_DEVICES=0 \
  ../TripoSplat/.venv/bin/python examples/triposplat/run_e2e.py \
  ../TripoSplat/static/example_inputs/building_stone_house.webp \
  --binary ./builds/vulkan-cm2/bin/triposplat-vulkan \
  --checkpoints ../TripoSplat/ckpts \
  --output ../TripoSplat/e2e_outputs/building_stone_house_32768_vulkan \
  --steps 20 --guidance 3.0 --num-gaussians 32768 --seed 42
```

This writes `.ply`, `.splat`, and `-preprocessed.webp`. The Python process owns
only image preparation, random-number generation and control flow. Standard
safetensors archives connect the Vulkan worker stages. The fixed scrambled
Sobol position grid is bundled in `assets/flow_positions.safetensors`, so the
runtime remains Torch-free.

On the A100 test system the command above completed in 24.443 seconds including
checkpoint loads and wrote exactly 32,768 finite PLY records and 32,768 SPLAT
records. An RGBA image with real alpha skips BiRefNet just like the reference;
an RGB image runs the full Vulkan BiRefNet background-removal graph.

## Torch/Vulkan parity

The fixture generators live in the neighbouring `TripoSplat` directory:

- `make_flow_parity.py`
- `make_dino_parity.py`
- `make_vae_parity.py`
- `make_gs_parity.py`
- `make_octree_parity.py`
- `make_deform_parity.py`
- `make_swin_parity.py`
- `make_biref_parity.py`
- `make_sample_parity.py`

Example full-flow comparison:

```sh
cd ../TripoSplat
./.venv/bin/python make_flow_parity.py \
  --output flow_camera_b24.safetensors --blocks 24

cd ../ggml
GGML_VK_VISIBLE_DEVICES=0 \
  ./build-vulkan/bin/triposplat-vulkan \
  --flow-parity \
  ../TripoSplat/ckpts/diffusion_models/triposplat_fp16.safetensors \
  ../TripoSplat/flow_camera_b24.safetensors 24
```

All `*-parity` modes validate that every graph node is supported by the Vulkan
backend before allocation and execution. Thus a passing test cannot silently
fall back to CPU. `llvmpipe` is useful for correctness testing when no hardware
Vulkan device is present, although it is much slower than a discrete GPU.

Representative full-network results against PyTorch (llvmpipe, F16 checkpoint
weights) are:

| Network | MAE | max error | cosine |
|---|---:|---:|---:|
| Flow, 24 blocks, latent | 0.000458 | 0.00207 | 0.99999975 |
| Flow, 24 blocks, camera | 0.000992 | 0.00202 | 0.99999986 |
| DINOv3, 32 blocks | 0.00454 | 0.07897 | 0.99974256 |
| Flux2 VAE encoder | 0.00800 | 0.03557 | 0.99992460 |
| Gaussian decoder, 16 blocks | 0.000104 | 0.000721 | 1.00000000 |
| Octree decoder, 4 blocks | 0.00153 | 0.00449 | 0.99999992 |
| Full BiRefNet | 1.29e-7 | 1.25e-6 | 0.99992058 |
| Deform conv, 7x7 | 0.000327 | 0.00128 | 0.99999995 |

`--sample-parity` additionally exercises graph reuse and host-side CFG Euler
updates. Very short synthetic token sequences are intentionally a numerical
stress case: small per-block F16/flash-attention differences compound over 24
blocks and are amplified by CFG. The full one-pass component checks above are
the authoritative layout and implementation parity tests.

## Compute-only benchmark

Set `TRIPOSPLAT_BENCH_ITERS` to execute an already allocated graph repeatedly
and print total and per-iteration compute time, excluding checkpoint loading and
graph construction:

```sh
TRIPOSPLAT_BENCH_ITERS=10 TRIPOSPLAT_BENCH_WARMUP=3 \
  GGML_VK_VISIBLE_DEVICES=0 \
  ./build-vulkan/bin/triposplat-vulkan \
  --flow-parity \
  ../TripoSplat/ckpts/diffusion_models/triposplat_fp16.safetensors \
  ../TripoSplat/flow_camera_b24.safetensors 24
```

The matching warmed-up Torch benchmark is `../TripoSplat/benchmark_torch.py`;
pass `--device cuda` to benchmark the CUDA GPU implementation. It supports
`flow`, `dino`, `vae`, `gs`, and `octree`.

### Cooperative matrix 1 vs 2

`GGML_VULKAN_DISABLE_COOPMAT2` creates a controlled CM1 build with the same
Vulkan SDK and all other detected shader extensions as CM2:

```sh
cmake -S . -B build-vulkan-cm1-sdk -G Ninja \
  -DGGML_VULKAN=ON -DGGML_CPU=OFF -DGGML_BUILD_EXAMPLES=ON \
  -DGGML_VULKAN_DISABLE_COOPMAT2=ON \
  -DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" \
  -DVulkan_GLSLC_EXECUTABLE="$VULKAN_SDK/bin/glslc"

cmake -S . -B build-vulkan-cm2 -G Ninja \
  -DGGML_VULKAN=ON -DGGML_CPU=OFF -DGGML_BUILD_EXAMPLES=ON \
  -DVulkan_INCLUDE_DIR="$VULKAN_SDK/include" \
  -DVulkan_GLSLC_EXECUTABLE="$VULKAN_SDK/bin/glslc"
```

On an NVIDIA A100-SXM4-40GB with Vulkan SDK 1.4.350.1, warmed compute-only
production fixtures measured:

| Component | CM1, ms | CM2, ms | CM2 change |
|---|---:|---:|---:|
| Flow, 24 blocks | 1128.246 | 1288.785 | +14.2% |
| DINO, 32 layers, 1024x1024 | 303.534 | 334.115 | +10.1% |
| Flux2 VAE, 1024x1024 | 370.243 | 452.668 | +22.3% |
| Gaussian decoder, 16 blocks | 565.668 | 597.680 | +5.7% |

The startup device line is also an execution-path check: the controlled builds
report `matrix cores: KHR_coopmat` and `matrix cores: NV_coopmat2`,
respectively.

After GPU-timestamp profiling and enabling the measured optimizations by
default, the same A100 CM2 build measures:

| Component | Torch CUDA | Optimized Vulkan | Vulkan / Torch | Vulkan speedup vs original CM2 |
|---|---:|---:|---:|---:|
| Flow, FP16 | 263.977 ms | 484.000 ms | 1.83x | 2.64x |
| DINO, BF16 | 100.293 ms | 174.436 ms | 1.74x | 1.93x |
| Flux2 VAE, BF16 | 103.928 ms | 197.916 ms | 1.90x | 2.29x |
| Gaussian decoder, FP16 | 129.937 ms | 192.903 ms | 1.48x | 3.07x |

These are compute-only runs with three warmups and ten measured iterations.
The optimized path keeps flash-attention K/V in F16, specializes D=64 RMSNorm
for one 32-lane subgroup, avoids materialized broadcast tensors, fuses the
DINO rotary embedding, and keeps the VAE convolution/GroupNorm pipeline in
F16. `TRIPOSPLAT_ATTN_KV_TYPE=f32` restores F32 K/V for diagnostics, while
`TRIPOSPLAT_VAE_PERSISTENT_F16=0` restores the original VAE convolution and
activation types. Set both variables to restore the complete original VAE
baseline, including F32 attention K/V.
