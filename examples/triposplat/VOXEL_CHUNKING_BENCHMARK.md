# Memory-chunked voxel conversion benchmark

This benchmark compares the monolithic TSVOXEL v2 converter at commit
`6ccc53a3` with the memory-chunked implementation in this change.

## Method

- Input: `output/result.ply` (32,768 Gaussians)
- GPU: NVIDIA A100-SXM4-40GB, Vulkan device 0
- Driver: NVIDIA 580.159.03
- Parameters: converter defaults, including 10 integration samples per axis
- Timing: median of three process runs
- `conversion` includes GPU work, readback, CPU sorting, bitset construction,
  and RGB8 quantization. It excludes output-file writing and one-time Vulkan
  setup, which the CLI reports separately.
- `GPU` is Vulkan timestamp-query time.
- `total allocation` is the conservative peak of all converter Vulkan memory
  allocations, including device-local, host-visible staging, and readback
  memory.
- `device-local` is the subset allocated from Vulkan memory types carrying
  `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

## Results

| Grid | Old conversion | Chunked conversion | Chunked/old | Old GPU | Chunked GPU | Old total allocation | Chunked total allocation | Chunked device-local | Chunks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 13.505 ms | 20.294 ms | 1.503x | 1.570 ms | 1.579 ms | 8.62 MiB | 8.62 MiB | 6.25 MiB | 1 |
| 64³ | 24.841 ms | 36.415 ms | 1.466x | 3.866 ms | 3.884 ms | 17.19 MiB | 17.19 MiB | 14.13 MiB | 1 |
| 128³ | 102.071 ms | 104.914 ms | 1.028x | 19.277 ms | 19.183 ms | 85.51 MiB | 85.51 MiB | 77.13 MiB | 1 |
| 256³ | 694.719 ms | 726.882 ms | 1.046x | 97.422 ms | 118.136 ms | 631.56 MiB | 631.56 MiB | 581.13 MiB | 1 |
| 512³ | 5,386.336 ms | 5,371.938 ms | 0.997x | 644.615 ms | 673.634 ms | 4,998.75 MiB | 1,314.84 MiB | 1,157.13 MiB | 4 |
| 1024³ | fails | 41,969.840 ms | n/a | fails | 4,745.617 ms | fails | 1,365.05 MiB | 1,157.13 MiB | 32 |

At 512³, chunking reduces the measured peak of all Vulkan allocations by
73.7% while preserving conversion speed. At 1024³, the measured peaks are
1.333 GiB total and 1.130 GiB device-local. Both are below the 3 GiB target;
the conservative total has 55.6% headroom.

An independent 1024³ run polled `nvidia-smi` while the converter was active.
Device 0 rose from 4,813 MiB (unrelated baseline use) to 6,004 MiB, a
1,191 MiB increment including Vulkan/NVIDIA driver overhead. This is within
34 MiB of the converter's 1,157.13 MiB device-local allocation counter and
leaves 61.2% headroom below the 3 GiB target.

The old 1024³ path exits with
`Gaussian AABB pair count exceeds uint32`. Even without that limit, its two
dense buffers alone would require 20 GiB for accumulation and 16 GiB for
records. The chunked path keeps those buffers at most 33,554,432 voxels
and partitions both voxel indices and candidate-pair indexing.

## Result contract

The following comparisons passed:

- N=32, 64, and 128 against the Python/Kaolin NPZ reference: exact occupancy,
  RGB8 maximum byte error 1, and the existing decoded-color error limits.
- Forced 8-chunk N=32/64/128 conversions against the normal conversion:
  exact occupancy and RGB8 maximum byte error 1.
- N=256 and N=512 against the monolithic C++ converter: exact occupancy,
  RGB8 maximum byte error 1. The RGB8 mismatch rates were respectively
  `1.16114705e-6` and `1.0746237e-6`.
- N=1024 passed the streaming TSVOXEL structure, payload-size, bitset-padding,
  and occupancy-population contract.

Small RGB8 differences are expected because floating-point compare-and-swap
atomics may accumulate contributors in a different order. Occupancy remains
exact.

Both the native Linux build and the MinGW-w64 cross-build of
`triposplat-vulkan` completed successfully.
