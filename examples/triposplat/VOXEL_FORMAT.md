# TSVOXEL v1

`TSVOXEL` is a dependency-free sparse voxel file written by
`triposplat-vulkan voxelize`. All integers and IEEE-754 floats are
little-endian. The file consists of one fixed-size header followed by sparse
voxel records:

```text
+----------------------+ 0
| 128-byte v1 header   |
+----------------------+ 128
| record 0 (16 bytes)  |
| record 1 (16 bytes)  |
| ...                  |
+----------------------+
```

## Header

| Offset | Bytes | Type | Field | v1 value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | char[8] | magic | `TSVOXEL\0` |
| 8 | 4 | uint32 | version | `1` |
| 12 | 4 | uint32 | header_bytes | `128` |
| 16 | 4 | uint32 | resolution | Cubic grid side length |
| 20 | 4 | uint32 | axis_order | `0`: dense view is `[z,y,x]` |
| 24 | 4 | uint32 | color_type | `1`: linear RGB float32 |
| 28 | 4 | uint32 | record_bytes | `16` |
| 32 | 8 | uint64 | occupied_count | Number of occupied voxels |
| 40 | 8 | uint64 | record_count | Number of following records |
| 48 | 12 | float32[3] | origin | World-space minimum corner of voxel `(0,0,0)` |
| 60 | 4 | float32 | voxel_size | World-space cube edge length |
| 64 | 4 | float32 | iso | Gaussian ellipsoid isovalue |
| 68 | 4 | float32 | opacity_threshold | Minimum accumulated occupancy |
| 72 | 4 | float32 | tolerance | Minimum Gaussian scale relative to a voxel |
| 76 | 4 | float32 | color_weight_power | Exponent used for color weights |
| 80 | 4 | uint32 | integration_steps | Samples per voxel axis |
| 84 | 4 | uint32 | flags | Bit 0: little-endian; bit 1: unordered records |
| 88 | 8 | uint64 | source_gaussian_count | Number of input Gaussians |
| 96 | 8 | uint64 | payload_bytes | `record_count * record_bytes` |
| 104 | 24 | uint64[3] | reserved | Zero |

Readers must use `header_bytes` to find the first record, reject unknown
required encodings, and tolerate future headers larger than 128 bytes.

## Sparse record

| Offset | Bytes | Type | Field |
|---:|---:|---|---|
| 0 | 4 | uint32 | linear_index |
| 4 | 12 | float32[3] | RGB |

For resolution `N`, decode an index as:

```text
x = linear_index % N
y = (linear_index / N) % N
z = linear_index / (N*N)
```

Equivalently, `linear_index = x + N * (y + N * z)`. A dense occupancy or
color array therefore has shape `[N,N,N]` or `[N,N,N,3]` in `z,y,x` order.
Records are unique but are not sorted because Vulkan workgroups append them
concurrently.

The world-space cube for `(x,y,z)` is:

```text
minimum = origin + voxel_size * (x,y,z)
maximum = minimum + voxel_size
center  = minimum + 0.5 * voxel_size
```

## Grid fit

Let `p_min` and `p_max` be the component-wise extrema of the input Gaussian
centers and let `span = max(p_max - p_min)`. The converter preserves aspect
ratio with cubic voxels:

```text
voxel_size = span / (N - 1)
origin     = p_min - 0.5 * voxel_size
```

Thus each minimum Gaussian-center coordinate lies at the center of index zero
on that axis. Shorter axes retain their aspect ratio and leave unused space at
the positive side of the cubic grid.

## Minimal C++ declarations

The declarations below describe the byte layout; do not write compiler-padded
versions without the packing assertion.

```cpp
#pragma pack(push, 1)
struct tsvoxel_header_v1 {
    char magic[8];
    uint32_t version, header_bytes;
    uint32_t resolution, axis_order, color_type, record_bytes;
    uint64_t occupied_count, record_count;
    float origin[3], voxel_size;
    float iso, opacity_threshold, tolerance, color_weight_power;
    uint32_t integration_steps, flags;
    uint64_t source_gaussian_count, payload_bytes, reserved[3];
};
#pragma pack(pop)
static_assert(sizeof(tsvoxel_header_v1) == 128);

struct tsvoxel_record_v1 {
    uint32_t linear_index;
    float rgb[3];
};
static_assert(sizeof(tsvoxel_record_v1) == 16);
```
