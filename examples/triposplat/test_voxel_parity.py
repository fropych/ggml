#!/usr/bin/env python3
"""Validate surface-only TSVOXEL v2 against volume references."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8s6I2Q3f5f2I5Q")
BASE_FLAGS = 1 | 2 | 4
SURFACE_SHELL_FLAG = 8
REQUIRED_FLAGS = BASE_FLAGS | SURFACE_SHELL_FLAG


@dataclass(frozen=True)
class VoxelFile:
    resolution: int
    occupied_count: int
    origin: np.ndarray
    voxel_size: float
    iso: float
    opacity_threshold: float
    tolerance: float
    color_weight_power: float
    integration_steps: int
    source_gaussian_count: int
    flags: int
    occupancy: np.ndarray
    colors_rgb8: np.ndarray


def validate_structure_streaming(path: Path) -> None:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError(f"{path}: truncated header")
        fields = HEADER.unpack(header)
        (
            magic,
            version,
            header_bytes,
            resolution,
            axis_order,
            color_type,
            record_bytes,
            occupied_count,
            record_count,
            _origin_x,
            _origin_y,
            _origin_z,
            _voxel_size,
            _iso,
            _opacity_threshold,
            _tolerance,
            _color_weight_power,
            _integration_steps,
            flags,
            _source_gaussian_count,
            payload_bytes,
            occupancy_bytes,
            color_bytes,
            reserved,
        ) = fields
        if magic != b"TSVOXEL\0":
            raise ValueError(f"{path}: invalid magic {magic!r}")
        if version != 2 or header_bytes < HEADER.size:
            raise ValueError(
                f"{path}: unsupported version/header "
                f"{version}/{header_bytes}"
            )
        if resolution < 2 or resolution & (resolution - 1):
            raise ValueError(
                f"{path}: resolution is not a power of two"
            )
        if axis_order != 0 or color_type != 2 or record_bytes != 3:
            raise ValueError(
                f"{path}: unsupported axis/color encoding"
            )
        if flags != REQUIRED_FLAGS:
            raise ValueError(f"{path}: unsupported v2 flags 0x{flags:x}")
        if occupied_count != record_count:
            raise ValueError(
                f"{path}: occupied_count != record_count"
            )
        voxel_count = resolution**3
        expected_occupancy_bytes = (voxel_count + 7) // 8
        if occupancy_bytes != expected_occupancy_bytes:
            raise ValueError(
                f"{path}: occupancy bytes {occupancy_bytes} "
                f"!= {expected_occupancy_bytes}"
            )
        if color_bytes != record_count * record_bytes:
            raise ValueError(f"{path}: inconsistent color byte count")
        if payload_bytes != occupancy_bytes + color_bytes:
            raise ValueError(f"{path}: inconsistent payload size")
        if file_size != header_bytes + payload_bytes:
            raise ValueError(
                f"{path}: file size {file_size} "
                f"!= {header_bytes + payload_bytes}"
            )
        if reserved:
            raise ValueError(f"{path}: non-zero v2 reserved field")

        stream.seek(header_bytes)
        population = 0
        remaining = occupancy_bytes
        last_byte = 0
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{path}: truncated occupancy bitset")
            population += sum(value.bit_count() for value in block)
            last_byte = block[-1]
            remaining -= len(block)
        trailing_bits = voxel_count & 7
        if trailing_bits and last_byte & ~((1 << trailing_bits) - 1):
            raise ValueError(
                f"{path}: non-zero occupancy padding bits"
            )
        if population != occupied_count:
            raise ValueError(
                f"{path}: bitset contains {population} occupied "
                f"voxels, header says {occupied_count}"
            )
        plane_bits = resolution * resolution

        def read_plane(z: int) -> np.ndarray:
            bit_begin = z * plane_bits
            bit_end = bit_begin + plane_bits
            byte_begin = bit_begin // 8
            byte_end = (bit_end + 7) // 8
            stream.seek(header_bytes + byte_begin)
            packed = stream.read(byte_end - byte_begin)
            if len(packed) != byte_end - byte_begin:
                raise ValueError(f"{path}: truncated occupancy bitset")
            bits = np.unpackbits(
                np.frombuffer(packed, dtype=np.uint8), bitorder="little"
            )
            offset = bit_begin - byte_begin * 8
            return bits[offset:offset + plane_bits].reshape(
                resolution, resolution
            ).astype(bool)

        if resolution > 2:
            previous = read_plane(0)
            current = read_plane(1)
            for z in range(1, resolution - 1):
                following = read_plane(z + 1)
                interior = current & previous & following
                interior[0, :] = False
                interior[-1, :] = False
                interior[:, 0] = False
                interior[:, -1] = False
                interior[1:, :] &= current[:-1, :]
                interior[:-1, :] &= current[1:, :]
                interior[:, 1:] &= current[:, :-1]
                interior[:, :-1] &= current[:, 1:]
                enclosed = int(np.count_nonzero(interior))
                if enclosed:
                    raise ValueError(
                        f"{path}: Z plane {z} contains {enclosed} "
                        "fully enclosed voxels"
                    )
                previous, current = current, following
    print(
        f"N={resolution}: streaming structure valid, "
        f"occupied={occupied_count}, bytes={file_size}"
    )


def load_tsvoxel(
    path: Path, *, require_surface_shell: bool = True
) -> VoxelFile:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    fields = HEADER.unpack_from(data)
    (
        magic,
        version,
        header_bytes,
        resolution,
        axis_order,
        color_type,
        record_bytes,
        occupied_count,
        record_count,
        origin_x,
        origin_y,
        origin_z,
        voxel_size,
        iso,
        opacity_threshold,
        tolerance,
        color_weight_power,
        integration_steps,
        flags,
        source_gaussian_count,
        payload_bytes,
        occupancy_bytes,
        color_bytes,
        reserved,
    ) = fields

    if magic != b"TSVOXEL\0":
        raise ValueError(f"{path}: invalid magic {magic!r}")
    if version != 2 or header_bytes < HEADER.size:
        raise ValueError(
            f"{path}: unsupported version/header {version}/{header_bytes}"
        )
    if resolution < 2 or resolution & (resolution - 1):
        raise ValueError(f"{path}: resolution is not a power of two")
    if axis_order != 0 or color_type != 2 or record_bytes != 3:
        raise ValueError(f"{path}: unsupported axis/color encoding")
    allowed_flags = (
        {REQUIRED_FLAGS}
        if require_surface_shell
        else {BASE_FLAGS, REQUIRED_FLAGS}
    )
    if flags not in allowed_flags:
        raise ValueError(f"{path}: unsupported v2 flags 0x{flags:x}")
    if occupied_count != record_count:
        raise ValueError(f"{path}: occupied_count != record_count")
    voxel_count = resolution**3
    expected_occupancy_bytes = (voxel_count + 7) // 8
    expected_color_bytes = record_count * record_bytes
    if occupancy_bytes != expected_occupancy_bytes:
        raise ValueError(
            f"{path}: occupancy bytes {occupancy_bytes} "
            f"!= {expected_occupancy_bytes}"
        )
    if color_bytes != expected_color_bytes:
        raise ValueError(
            f"{path}: color bytes {color_bytes} != {expected_color_bytes}"
        )
    if payload_bytes != occupancy_bytes + color_bytes:
        raise ValueError(f"{path}: inconsistent payload size")
    if len(data) != header_bytes + payload_bytes:
        raise ValueError(
            f"{path}: file size {len(data)} != {header_bytes + payload_bytes}"
        )
    if reserved:
        raise ValueError(f"{path}: non-zero v2 reserved field")

    bitset_end = header_bytes + occupancy_bytes
    bitset = np.frombuffer(
        data, dtype=np.uint8, count=occupancy_bytes, offset=header_bytes
    )
    trailing_bits = voxel_count & 7
    if trailing_bits:
        padding_mask = np.uint8(0xFF ^ ((1 << trailing_bits) - 1))
        if bitset[-1] & padding_mask:
            raise ValueError(f"{path}: non-zero occupancy padding bits")
    occupancy_flat = np.unpackbits(
        bitset, bitorder="little", count=voxel_count
    ).astype(bool)
    actual_occupied = int(np.count_nonzero(occupancy_flat))
    if actual_occupied != occupied_count:
        raise ValueError(
            f"{path}: bitset contains {actual_occupied} occupied voxels, "
            f"header says {occupied_count}"
        )

    sparse_colors = np.frombuffer(
        data, dtype=np.uint8, count=color_bytes, offset=bitset_end
    ).reshape((record_count, 3))
    colors = np.zeros((voxel_count, 3), dtype=np.uint8)
    colors[occupancy_flat] = sparse_colors
    shape = (resolution, resolution, resolution)
    return VoxelFile(
        resolution=resolution,
        occupied_count=occupied_count,
        origin=np.array([origin_x, origin_y, origin_z], dtype=np.float32),
        voxel_size=voxel_size,
        iso=iso,
        opacity_threshold=opacity_threshold,
        tolerance=tolerance,
        color_weight_power=color_weight_power,
        integration_steps=integration_steps,
        source_gaussian_count=source_gaussian_count,
        flags=flags,
        occupancy=occupancy_flat.reshape(shape),
        colors_rgb8=colors.reshape((*shape, 3)),
    )


def metadata(reference: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in reference.files:
        return {}
    return json.loads(str(reference["metadata_json"].item()))


def check_close(label: str, actual: float, expected: float, atol: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: {actual:.9g} != {expected:.9g}")


def quantize_rgb8(colors: np.ndarray) -> np.ndarray:
    scaled = (
        np.clip(colors, np.float32(0.0), np.float32(1.0))
        * np.float32(255.0)
        + np.float32(0.5)
    )
    return np.floor(scaled).astype(np.uint8)


def surface_shell(occupancy: np.ndarray) -> np.ndarray:
    """Retain occupied cells with at least one empty/out-of-grid face neighbour."""
    occupancy = np.asarray(occupancy, dtype=bool)
    interior = occupancy.copy()
    interior[0, :, :] = False
    interior[-1, :, :] = False
    interior[:, 0, :] = False
    interior[:, -1, :] = False
    interior[:, :, 0] = False
    interior[:, :, -1] = False
    interior[1:, :, :] &= occupancy[:-1, :, :]
    interior[:-1, :, :] &= occupancy[1:, :, :]
    interior[:, 1:, :] &= occupancy[:, :-1, :]
    interior[:, :-1, :] &= occupancy[:, 1:, :]
    interior[:, :, 1:] &= occupancy[:, :, :-1]
    interior[:, :, :-1] &= occupancy[:, :, 1:]
    return occupancy & ~interior


def assert_surface_shell(occupancy: np.ndarray) -> None:
    interior_count = int(np.count_nonzero(occupancy & ~surface_shell(occupancy)))
    if interior_count:
        raise AssertionError(
            f"output contains {interior_count} fully enclosed voxels"
        )


def compare_tsvoxel(
    actual: VoxelFile,
    baseline: VoxelFile,
    max_rgb8_mismatch_rate: float,
    max_rgb8_byte_error: int,
) -> None:
    scalar_fields = (
        "resolution",
        "voxel_size",
        "iso",
        "opacity_threshold",
        "tolerance",
        "color_weight_power",
        "integration_steps",
        "source_gaussian_count",
    )
    for field in scalar_fields:
        if getattr(actual, field) != getattr(baseline, field):
            raise AssertionError(
                f"baseline {field}: {getattr(actual, field)!r} "
                f"!= {getattr(baseline, field)!r}"
            )
    if not np.array_equal(actual.origin, baseline.origin):
        raise AssertionError("baseline origin differs")
    expected_occupancy = (
        baseline.occupancy
        if baseline.flags & SURFACE_SHELL_FLAG
        else surface_shell(baseline.occupancy)
    )
    expected_count = int(np.count_nonzero(expected_occupancy))
    if actual.occupied_count != expected_count:
        raise AssertionError(
            f"surface occupied_count: {actual.occupied_count} != "
            f"{expected_count}"
        )
    if not np.array_equal(actual.occupancy, expected_occupancy):
        xor_count = int(np.count_nonzero(
            actual.occupancy ^ expected_occupancy
        ))
        raise AssertionError(
            f"baseline occupancy differs at {xor_count} voxels"
        )

    byte_error = np.abs(
        actual.colors_rgb8[actual.occupancy].astype(np.int16)
        - baseline.colors_rgb8[expected_occupancy].astype(np.int16)
    )
    mismatch_count = int(np.count_nonzero(byte_error))
    mismatch_rate = (
        mismatch_count / byte_error.size if byte_error.size else 0.0
    )
    maximum_byte_error = int(byte_error.max()) if byte_error.size else 0
    print(
        f"baseline TSVOX: occupancy exact, RGB8 mismatches="
        f"{mismatch_count}/{byte_error.size} ({mismatch_rate:.9g}), "
        f"byte max={maximum_byte_error}"
    )
    if mismatch_rate > max_rgb8_mismatch_rate:
        raise AssertionError(
            f"baseline RGB8 mismatch rate {mismatch_rate:.9g} "
            f"> {max_rgb8_mismatch_rate:.9g}"
        )
    if maximum_byte_error > max_rgb8_byte_error:
        raise AssertionError(
            f"baseline RGB8 max byte error {maximum_byte_error} "
            f"> {max_rgb8_byte_error}"
        )


def compare(
    actual: VoxelFile,
    reference_path: Path,
    max_rgb8_mismatch_rate: float,
    max_rgb8_byte_error: int,
    max_color_mae: float,
    max_color_error: float,
) -> None:
    with np.load(reference_path, allow_pickle=False) as reference:
        expected_occupancy = np.asarray(reference["occupancy"], dtype=bool)
        expected_colors = np.asarray(reference["colors"], dtype=np.float32)
        expected_origin = np.asarray(reference["origin"], dtype=np.float32)
        expected_voxel_size = float(reference["voxel_size"])
        expected_axis_order = str(reference["axis_order"].item())
        expected_metadata = metadata(reference)

    expected_shape = (actual.resolution,) * 3
    if expected_occupancy.shape != expected_shape:
        raise AssertionError(
            f"reference occupancy shape {expected_occupancy.shape} != {expected_shape}"
        )
    if expected_colors.shape != (*expected_shape, 3):
        raise AssertionError(
            f"reference colors shape {expected_colors.shape} is invalid"
        )
    if expected_axis_order != "zyx":
        raise AssertionError(f"reference axis_order {expected_axis_order!r} != 'zyx'")
    if not np.array_equal(actual.origin, expected_origin):
        raise AssertionError(
            f"origin differs: {actual.origin.tolist()} != {expected_origin.tolist()}"
        )
    if np.float32(actual.voxel_size) != np.float32(expected_voxel_size):
        raise AssertionError(
            f"voxel_size differs: {actual.voxel_size} != {expected_voxel_size}"
        )

    parameter_fields = (
        "iso",
        "opacity_threshold",
        "tolerance",
        "color_weight_power",
        "integration_steps",
    )
    for field in parameter_fields:
        if field not in expected_metadata:
            continue
        actual_value = getattr(actual, field)
        expected_value = expected_metadata[field]
        if field == "integration_steps":
            if actual_value != int(expected_value):
                raise AssertionError(
                    f"{field}: {actual_value} != {int(expected_value)}"
                )
        else:
            check_close(field, actual_value, float(expected_value), 1e-6)

    expected_occupancy = surface_shell(expected_occupancy)
    intersection = np.count_nonzero(actual.occupancy & expected_occupancy)
    union = np.count_nonzero(actual.occupancy | expected_occupancy)
    xor_count = np.count_nonzero(actual.occupancy ^ expected_occupancy)
    iou = intersection / union if union else 1.0
    if xor_count:
        raise AssertionError(
            f"occupancy differs: xor={xor_count}, IoU={iou:.9f}"
        )

    actual_rgb8 = actual.colors_rgb8[expected_occupancy]
    reference_colors = expected_colors[expected_occupancy]
    expected_rgb8 = quantize_rgb8(reference_colors)
    byte_error = np.abs(
        actual_rgb8.astype(np.int16) - expected_rgb8.astype(np.int16)
    )
    mismatch_count = int(np.count_nonzero(byte_error))
    mismatch_rate = (
        mismatch_count / byte_error.size if byte_error.size else 0.0
    )
    maximum_byte_error = int(byte_error.max()) if byte_error.size else 0

    decoded_colors = actual_rgb8.astype(np.float32) / np.float32(255.0)
    color_error = np.abs(decoded_colors - reference_colors)
    mae = float(color_error.mean()) if color_error.size else 0.0
    maximum = float(color_error.max()) if color_error.size else 0.0
    percentile_99 = (
        float(np.percentile(color_error, 99.0)) if color_error.size else 0.0
    )
    print(
        f"N={actual.resolution}: occupied={actual.occupied_count}, "
        f"IoU={iou:.9f}, xor={xor_count}, "
        f"RGB8 mismatches={mismatch_count}/{byte_error.size} "
        f"({mismatch_rate:.9g}), byte max={maximum_byte_error}, "
        f"decoded MAE={mae:.9g}, p99={percentile_99:.9g}, "
        f"max={maximum:.9g}"
    )
    if mismatch_rate > max_rgb8_mismatch_rate:
        raise AssertionError(
            f"RGB8 mismatch rate {mismatch_rate:.9g} "
            f"> {max_rgb8_mismatch_rate:.9g}"
        )
    if maximum_byte_error > max_rgb8_byte_error:
        raise AssertionError(
            f"RGB8 max byte error {maximum_byte_error} "
            f"> {max_rgb8_byte_error}"
        )
    if mae > max_color_mae:
        raise AssertionError(f"decoded RGB MAE {mae:.9g} > {max_color_mae:.9g}")
    if maximum > max_color_error:
        raise AssertionError(
            f"decoded RGB max error {maximum:.9g} > {max_color_error:.9g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsvoxel", type=Path)
    parser.add_argument(
        "reference_npz",
        type=Path,
        nargs="?",
        help="optional Python/Kaolin NPZ reference",
    )
    parser.add_argument(
        "--baseline-tsvoxel",
        type=Path,
        help="compare exact occupancy and bounded RGB8 with a baseline TSVOXEL",
    )
    parser.add_argument("--max-rgb8-mismatch-rate", type=float, default=5e-4)
    parser.add_argument("--max-rgb8-byte-error", type=int, default=1)
    parser.add_argument("--max-color-mae", type=float, default=1.05e-3)
    parser.add_argument("--max-color-error", type=float, default=2.1e-3)
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="stream-validate the file without dense NumPy materialization",
    )
    args = parser.parse_args()
    if (
        not args.structure_only
        and args.reference_npz is None
        and args.baseline_tsvoxel is None
    ):
        parser.error(
            "provide reference_npz, --baseline-tsvoxel, "
            "and/or --structure-only"
        )
    try:
        if args.structure_only:
            validate_structure_streaming(args.tsvoxel)
            if (
                args.reference_npz is None
                and args.baseline_tsvoxel is None
            ):
                print("PASS")
                return 0
        actual = load_tsvoxel(args.tsvoxel)
        assert_surface_shell(actual.occupancy)
        if args.baseline_tsvoxel is not None:
            baseline = load_tsvoxel(
                args.baseline_tsvoxel, require_surface_shell=False
            )
            compare_tsvoxel(
                actual,
                baseline,
                args.max_rgb8_mismatch_rate,
                args.max_rgb8_byte_error,
            )
        if args.reference_npz is not None:
            compare(
                actual,
                args.reference_npz,
                args.max_rgb8_mismatch_rate,
                args.max_rgb8_byte_error,
                args.max_color_mae,
                args.max_color_error,
            )
    except (AssertionError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
