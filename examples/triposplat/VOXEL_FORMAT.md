# TSVOXEL v2

`TSVOXEL` is a dependency-free voxel file written by
`triposplat-vulkan voxelize`. Version 2 stores occupancy as one bit per dense
voxel and stores one RGB8 triplet per occupied voxel. All header integers and
IEEE-754 floats are little-endian.

```text
+-----------------------------+ 0
| 128-byte v2 header          |
+-----------------------------+ header_bytes
| occupancy bitset            | ceil(N^3 / 8) bytes
+-----------------------------+
| RGB8 for every set bit      | occupied_count * 3 bytes
+-----------------------------+
```

## Header

| Offset | Bytes | Type | Field | v2 value or meaning |
|---:|---:|---|---|---|
| 0 | 8 | char[8] | magic | `TSVOXEL\0` |
| 8 | 4 | uint32 | version | `2` |
| 12 | 4 | uint32 | header_bytes | `128` |
| 16 | 4 | uint32 | resolution | Cubic grid side length `N` |
| 20 | 4 | uint32 | axis_order | `0`: dense view is `[z,y,x]` |
| 24 | 4 | uint32 | color_type | `2`: linear RGB UNORM8 |
| 28 | 4 | uint32 | record_bytes | `3`: bytes per RGB entry |
| 32 | 8 | uint64 | occupied_count | Number of set occupancy bits |
| 40 | 8 | uint64 | record_count | Number of RGB triplets; equals `occupied_count` |
| 48 | 12 | float32[3] | origin | World-space minimum corner of voxel `(0,0,0)` |
| 60 | 4 | float32 | voxel_size | World-space cube edge length |
| 64 | 4 | float32 | iso | Gaussian ellipsoid isovalue |
| 68 | 4 | float32 | opacity_threshold | Minimum accumulated occupancy |
| 72 | 4 | float32 | tolerance | Minimum Gaussian scale relative to a voxel |
| 76 | 4 | float32 | color_weight_power | Exponent used for color weights |
| 80 | 4 | uint32 | integration_steps | Samples per voxel axis |
| 84 | 4 | uint32 | flags | Required bits are described below |
| 88 | 8 | uint64 | source_gaussian_count | Number of input Gaussians |
| 96 | 8 | uint64 | payload_bytes | `occupancy_bytes + color_bytes` |
| 104 | 8 | uint64 | occupancy_bytes | `ceil(N^3 / 8)` |
| 112 | 8 | uint64 | color_bytes | `record_count * 3` |
| 120 | 8 | uint64 | reserved | Zero |

Required `flags` bits:

| Bit | Meaning |
|---:|---|
| 0 | Header values are little-endian |
| 1 | Occupancy is an LSB-first bitset |
| 2 | RGB entries follow set bits in increasing `linear_index` |

Readers must use `header_bytes` to find the bitset and reject files with
unknown required encodings. A future version may use a header larger than
128 bytes.

## Linear voxel order

For resolution `N`:

```text
linear_index = x + N * (y + N * z)

x = linear_index % N
y = (linear_index / N) % N
z = linear_index / (N*N)
```

A dense occupancy or color array therefore has shape `[N,N,N]` or
`[N,N,N,3]` in `z,y,x` order.

## Occupancy bitset

Voxel `i` is represented by one bit:

```text
byte_index = i / 8
bit_index  = i % 8
occupied   = (bitset[byte_index] & (1 << bit_index)) != 0
```

Bit zero is the least-significant bit of the first byte. If `N^3` is not a
multiple of eight, unused high bits of the last byte must be zero.

The number of set bits must equal both `occupied_count` and `record_count`.

## RGB order and quantization

There is no index beside each color. RGB entries correspond to occupied
voxels in increasing `linear_index`. A sequential reader keeps a color
cursor:

```cpp
size_t color_index = 0;
for (uint64_t voxel = 0; voxel < voxel_count; ++voxel) {
    if ((bitset[voxel >> 3] & (1u << (voxel & 7u))) == 0) {
        continue;
    }
    const uint8_t * rgb = colors + 3 * color_index++;
    // rgb belongs to this voxel
}
```

The converter retains float32 while integrating Gaussian contributions. Only
the finalized, clamped linear RGB is quantized on CPU:

```text
encoded = floor(clamp(linear_rgb, 0, 1) * 255 + 0.5)
decoded = encoded / 255
```

The maximum quantization error per channel is `0.5 / 255`, approximately
`0.0019608`.

## World-space mapping

The world-space cube for `(x,y,z)` is:

```text
minimum = origin + voxel_size * (x,y,z)
maximum = minimum + voxel_size
center  = minimum + 0.5 * voxel_size
```

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

## Minimal C++ header declaration

```cpp
#pragma pack(push, 1)
struct tsvoxel_header_v2 {
    char magic[8];
    uint32_t version, header_bytes;
    uint32_t resolution, axis_order, color_type, record_bytes;
    uint64_t occupied_count, record_count;
    float origin[3], voxel_size;
    float iso, opacity_threshold, tolerance, color_weight_power;
    uint32_t integration_steps, flags;
    uint64_t source_gaussian_count, payload_bytes;
    uint64_t occupancy_bytes, color_bytes, reserved;
};
#pragma pack(pop)
static_assert(sizeof(tsvoxel_header_v2) == 128);
```
