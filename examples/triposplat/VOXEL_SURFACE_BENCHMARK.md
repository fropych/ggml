# Surface-shell voxel conversion benchmark

This benchmark compares the filled-volume chunked TSVOXEL v2 converter at
commit `af6485de` with the mandatory six-face-neighbour surface filtering in
this change.

## Method

- Input: `output/result.ply` (32,768 Gaussians)
- GPU: NVIDIA A100-SXM4-40GB, Vulkan device 0
- Driver: NVIDIA 580.159.03
- Parameters: converter defaults, including 10 integration samples per axis
- Timing: median of three separate process runs
- `conversion` includes GPU work, readback, CPU sorting, bitset construction,
  and RGB8 quantization. It excludes output-file writing and one-time Vulkan
  setup, which the CLI reports separately.
- `process wall` includes process startup, Vulkan setup, conversion, and file
  writing, measured with `/usr/bin/time`.
- `GPU` is Vulkan timestamp-query time for preprocess, integration, and
  finalize dispatches.
- `total allocation` is the conservative peak of all converter Vulkan memory
  allocations, including device-local, staging, and readback memory.
- `device-local` is the subset allocated from Vulkan memory types carrying
  `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`.

No 1024³ run was made for this comparison.

## Time and output

| Grid | Volume voxels | Surface voxels | Removed | Volume conversion | Surface conversion | Speedup | Volume process wall | Surface process wall | Volume file | Surface file | File reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 7,800 | 3,923 | 49.7% | 19.528 ms | 13.129 ms | 1.487x | 0.40 s | 0.40 s | 27,624 B | 15,993 B | 42.1% |
| 64³ | 53,484 | 17,909 | 66.5% | 31.949 ms | 18.626 ms | 1.715x | 0.41 s | 0.39 s | 193,348 B | 86,623 B | 55.2% |
| 128³ | 401,992 | 77,303 | 80.8% | 98.612 ms | 45.174 ms | 2.183x | 0.47 s | 0.43 s | 1,468,248 B | 494,181 B | 66.3% |
| 256³ | 3,157,797 | 342,716 | 89.1% | 655.122 ms | 176.908 ms | 3.703x | 1.03 s | 0.54 s | 11,570,671 B | 3,125,428 B | 73.0% |
| 512³ | 25,125,074 | 1,457,821 | 94.2% | 4,947.369 ms | 1,049.402 ms | 4.714x | 5.42 s | 1.46 s | 92,152,566 B | 21,150,807 B | 77.0% |

The Gaussian integration work is fundamentally unchanged. Larger speedups
come from emitting, reading back, sorting, and quantizing far fewer records.
The surface test also reduces atomic contention in the finalize dispatch.

## GPU time and memory

| Grid | Volume GPU | Surface GPU | Volume total allocation | Surface total allocation | Reduction | Volume device-local | Surface device-local | Surface chunks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32³ | 1.566 ms | 1.567 ms | 8.62 MiB | 8.56 MiB | 0.7% | 6.25 MiB | 6.25 MiB | 1 |
| 64³ | 3.847 ms | 3.828 ms | 17.19 MiB | 16.65 MiB | 3.1% | 14.13 MiB | 14.13 MiB | 1 |
| 128³ | 19.301 ms | 19.039 ms | 85.51 MiB | 80.55 MiB | 5.8% | 77.13 MiB | 77.13 MiB | 1 |
| 256³ | 98.138 ms | 90.496 ms | 631.56 MiB | 588.60 MiB | 6.8% | 581.13 MiB | 581.13 MiB | 1 |
| 512³ | 665.629 ms | 573.342 ms | 1,314.84 MiB | 1,157.89 MiB | 11.9% | 1,157.13 MiB | 1,149.13 MiB | 5 |

At 512³, each allocation chunk still contains at most 33,554,432 integrated
voxels. Two of its 128 Z planes are halos, leaving at most 126 owned planes and
requiring five output chunks. The record buffer only needs capacity for owned
cells. Device-local memory therefore remains near the previous value, while
the much smaller readback allocation lowers the conservative total peak.

## Result contract

The updated `test_voxel_parity.py` requires the surface-shell header flag and
proves that no output cell has all six face neighbours occupied.

The following comparisons passed:

- N=32, 64, and 128 against the Python/Kaolin volume reference after an
  independent CPU six-neighbour surface derivation: exact occupancy, RGB8
  maximum byte error 1, and the existing decoded-color error limits.
- N=32, 64, 128, 256, and 512 against filled-volume TSVOXEL baselines after
  the same independent surface derivation: exact occupancy. RGB8 maximum byte
  error was 1; N=256 had 3 differing bytes out of 1,028,148 and N=512 had
  8 differing bytes out of 4,373,463.
- Forced N=32 conversion with four owned Z layers per chunk against the normal
  single-chunk conversion: exact occupancy and exact RGB8.
- N=256 and N=512 passed streaming header, payload, bitset-padding,
  population, and surface-shell invariant validation.

Small RGB8 differences are expected because floating-point compare-and-swap
atomics may accumulate contributors in a different order. Occupancy is exact.

Both the native Linux build/package and the MinGW-w64 Windows cross-build/
package completed successfully, including the embedded surface-filter SPIR-V
and this benchmark document.
