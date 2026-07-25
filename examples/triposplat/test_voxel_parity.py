#!/usr/bin/env python3
"""Validate a TSVOXEL file and compare it with the NumPy reference."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<8s6I2Q3f5f2I2Q3Q")
RECORD_DTYPE = np.dtype(
    [("linear_index", "<u4"), ("rgb", "<f4", (3,))], align=False
)


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
    occupancy: np.ndarray
    colors: np.ndarray


def load_tsvoxel(path: Path) -> VoxelFile:
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
        reserved_0,
        reserved_1,
        reserved_2,
    ) = fields

    if magic != b"TSVOXEL\0":
        raise ValueError(f"{path}: invalid magic {magic!r}")
    if version != 1 or header_bytes < HEADER.size:
        raise ValueError(
            f"{path}: unsupported version/header {version}/{header_bytes}"
        )
    if resolution < 2 or resolution & (resolution - 1):
        raise ValueError(f"{path}: resolution is not a power of two")
    if axis_order != 0 or color_type != 1 or record_bytes != RECORD_DTYPE.itemsize:
        raise ValueError(f"{path}: unsupported axis/color/record encoding")
    if not flags & 1:
        raise ValueError(f"{path}: little-endian flag is missing")
    if occupied_count != record_count:
        raise ValueError(f"{path}: occupied_count != record_count")
    if payload_bytes != record_count * record_bytes:
        raise ValueError(f"{path}: inconsistent payload size")
    if len(data) != header_bytes + payload_bytes:
        raise ValueError(
            f"{path}: file size {len(data)} != {header_bytes + payload_bytes}"
        )
    if reserved_0 or reserved_1 or reserved_2:
        raise ValueError(f"{path}: non-zero v1 reserved fields")

    records = np.frombuffer(
        data, dtype=RECORD_DTYPE, count=record_count, offset=header_bytes
    )
    indices = records["linear_index"].astype(np.int64)
    voxel_count = resolution**3
    if np.any(indices >= voxel_count):
        raise ValueError(f"{path}: out-of-range linear_index")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{path}: duplicate linear_index")
    if not np.isfinite(records["rgb"]).all():
        raise ValueError(f"{path}: non-finite color")
    if np.any(records["rgb"] < 0.0) or np.any(records["rgb"] > 1.0):
        raise ValueError(f"{path}: color outside [0,1]")

    occupancy = np.zeros(voxel_count, dtype=bool)
    colors = np.zeros((voxel_count, 3), dtype=np.float32)
    occupancy[indices] = True
    colors[indices] = records["rgb"]
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
        occupancy=occupancy.reshape(shape),
        colors=colors.reshape((*shape, 3)),
    )


def metadata(reference: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "metadata_json" not in reference.files:
        return {}
    return json.loads(str(reference["metadata_json"].item()))


def check_close(label: str, actual: float, expected: float, atol: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: {actual:.9g} != {expected:.9g}")


def compare(
    actual: VoxelFile,
    reference_path: Path,
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

    intersection = np.count_nonzero(actual.occupancy & expected_occupancy)
    union = np.count_nonzero(actual.occupancy | expected_occupancy)
    xor_count = np.count_nonzero(actual.occupancy ^ expected_occupancy)
    iou = intersection / union if union else 1.0
    if xor_count:
        raise AssertionError(
            f"occupancy differs: xor={xor_count}, IoU={iou:.9f}"
        )

    color_error = np.abs(
        actual.colors[expected_occupancy] - expected_colors[expected_occupancy]
    )
    mae = float(color_error.mean()) if color_error.size else 0.0
    maximum = float(color_error.max()) if color_error.size else 0.0
    percentile_99 = (
        float(np.percentile(color_error, 99.0)) if color_error.size else 0.0
    )
    print(
        f"N={actual.resolution}: occupied={actual.occupied_count}, "
        f"IoU={iou:.9f}, xor={xor_count}, RGB MAE={mae:.9g}, "
        f"p99={percentile_99:.9g}, max={maximum:.9g}"
    )
    if mae > max_color_mae:
        raise AssertionError(f"RGB MAE {mae:.9g} > {max_color_mae:.9g}")
    if maximum > max_color_error:
        raise AssertionError(
            f"RGB max error {maximum:.9g} > {max_color_error:.9g}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsvoxel", type=Path)
    parser.add_argument("reference_npz", type=Path)
    parser.add_argument("--max-color-mae", type=float, default=1e-6)
    parser.add_argument("--max-color-error", type=float, default=2e-4)
    args = parser.parse_args()
    try:
        actual = load_tsvoxel(args.tsvoxel)
        compare(
            actual,
            args.reference_npz,
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
