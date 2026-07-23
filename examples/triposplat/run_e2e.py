#!/usr/bin/env python3
"""TripoSplat image-to-splat host orchestrator for the ggml Vulkan worker.

All learned model stages are executed by ``triposplat-vulkan``.  This file only
implements image preparation, stochastic sampling, octree control flow and
serialization, mirroring the corresponding host code in ``triposplat.py``.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from safetensors.numpy import load_file, save_file


CANVAS = 1024
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
TRANSFORM = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)


def contiguous(values, dtype=np.float32):
    return np.ascontiguousarray(values, dtype=dtype)


def resize_bilinear_align_corners(values, output_height: int, output_width: int):
    """NumPy equivalent of F.interpolate(..., align_corners=True)."""
    source_height, source_width = values.shape[-2:]
    ys = np.linspace(0.0, source_height - 1, output_height, dtype=np.float32)
    xs = np.linspace(0.0, source_width - 1, output_width, dtype=np.float32)
    y0, x0 = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, source_height - 1), np.minimum(x0 + 1, source_width - 1)
    wy = (ys - y0).reshape((-1, 1))
    wx = (xs - x0).reshape((1, -1))
    top = values[..., y0[:, None], x0[None, :]] * (1.0 - wx) + values[..., y0[:, None], x1[None, :]] * wx
    bottom = values[..., y1[:, None], x0[None, :]] * (1.0 - wx) + values[..., y1[:, None], x1[None, :]] * wx
    return contiguous(top * (1.0 - wy) + bottom * wy)


def attention_mask(height: int, width: int, window: int = 12):
    hp = math.ceil(height / window) * window
    wp = math.ceil(width / window) * window
    shift = window // 2
    image = np.zeros((1, hp, wp, 1), dtype=np.float32)
    hs = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    ws = (slice(0, -window), slice(-window, -shift), slice(-shift, None))
    count = 0
    for h in hs:
        for w in ws:
            image[:, h, w, :] = count
            count += 1
    windows = image.reshape(1, hp // window, window, wp // window, window, 1)
    windows = windows.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window * window)
    delta = windows[:, None, :] - windows[:, :, None]
    return contiguous(np.where(delta != 0, -100.0, 0.0))


def biref_masks(size: int):
    tensors = {}
    for prefix, input_size in (("full_", size), ("half_", size // 2)):
        height = math.ceil(input_size / 4)
        width = height
        for stage in range(4):
            tensors[f"{prefix}mask{stage}"] = attention_mask(height, width)
            height, width = (height + 1) // 2, (width + 1) // 2
    return tensors


def dino_rope(height: int, width: int, dim: int = 64, base: float = 100.0):
    inv_freq = 1.0 / (base ** np.arange(0, 1, 4.0 / dim, dtype=np.float32))
    coords_h = np.arange(0.5, height, dtype=np.float32) / height
    coords_w = np.arange(0.5, width, dtype=np.float32) / width
    yy, xx = np.meshgrid(coords_h, coords_w, indexing="ij")
    coords = (2.0 * np.stack((yy, xx), axis=-1) - 1.0).reshape(-1, 2)
    angles = (2 * np.pi * coords[:, :, None] * inv_freq[None, None, :]).reshape(-1, dim // 2)
    angles = np.tile(angles, (1, 2))[None, None]
    return contiguous(np.cos(angles)), contiguous(np.sin(angles))


def position_freqs_v1():
    freq_dim = 1024 // 3 // 2
    exponential = np.arange(16, dtype=np.float32)
    remaining = freq_dim - exponential.size
    residual = np.arange(remaining, dtype=np.float32) / max(remaining, 1) * 16.0
    return contiguous(np.power(2.0, np.concatenate((exponential, residual))[:freq_dim]))


def position_freqs_v2():
    return contiguous(np.power(2.0, np.linspace(0.0, 10.0, 1024 // 3 // 2, dtype=np.float32)))


def sobol_positions(count: int = 8192):
    asset = Path(__file__).resolve().parent / "assets/flow_positions.safetensors"
    positions = load_file(asset)["positions"]
    if positions.shape != (1, count, 3):
        raise RuntimeError(f"invalid Flow position asset shape: {positions.shape}")
    return contiguous(positions)


def run_worker(binary: Path, mode: str, model: Path, source: Path, target: Path, *extra):
    command = [str(binary), mode, str(model), str(source), str(target), *(str(x) for x in extra)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resize_short_side(image: Image.Image, size: int):
    width, height = image.size
    scale = size / min(width, height)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def preprocess(image_path: Path, binary: Path, biref_weights: Path, work: Path, erode: int):
    image = resize_short_side(Image.open(image_path), CANVAS)
    has_alpha = image.mode == "RGBA" and np.asarray(image.getchannel(3), dtype=np.uint8).min() < 255
    if not has_alpha:
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        pixels = resize_bilinear_align_corners(pixels, CANVAS, CANVAS)[None]
        source = work / "biref-input.safetensors"
        target = work / "biref-output.safetensors"
        save_file({"pixels": contiguous((pixels - MEAN) / STD), **biref_masks(CANVAS)}, source)
        run_worker(binary, "--run-biref", biref_weights, source, target)
        alpha = load_file(target)["alpha"][0, 0]
        alpha = resize_bilinear_align_corners(alpha, rgb.height, rgb.width)
        alpha_image = Image.fromarray(np.uint8(np.clip(alpha, 0, 1) * 255), mode="L")
        image = rgb.copy()
        image.putalpha(alpha_image)
    else:
        image = image.convert("RGBA")
    if erode > 0:
        image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(2 * erode + 1)))
    alpha = np.asarray(image.getchannel(3))
    ys, xs = np.nonzero(alpha)
    if not len(xs):
        raise RuntimeError("background removal produced an empty foreground")
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * 1.2
    image = image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    image = image.resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)
    prepared = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    prepared.paste(image, mask=image.getchannel(3))
    return prepared


def encode(prepared: Image.Image, binary: Path, dino_weights: Path, vae_weights: Path,
           work: Path, rng: np.random.Generator):
    image = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    cosine, sine = dino_rope(CANVAS // 16, CANVAS // 16)
    dino_in, dino_out = work / "dino-input.safetensors", work / "dino-output.safetensors"
    save_file({"pixels": contiguous((image - MEAN) / STD), "rope_cos": cosine, "rope_sin": sine}, dino_in)
    run_worker(binary, "--run-dino", dino_weights, dino_in, dino_out)
    feature1 = contiguous(load_file(dino_out)["feature1"])
    feature1_mean = feature1.mean(axis=-1, keepdims=True)
    feature1_variance = np.mean((feature1 - feature1_mean) ** 2, axis=-1, keepdims=True)
    feature1 = contiguous((feature1 - feature1_mean) / np.sqrt(feature1_variance + 1e-5))

    vae_in, vae_out = work / "vae-input.safetensors", work / "vae-output.safetensors"
    noise = contiguous(rng.standard_normal((1, 32, CANVAS // 8, CANVAS // 8)))
    save_file({"image": contiguous(image * 2.0 - 1.0), "noise": noise}, vae_in)
    run_worker(binary, "--run-vae", vae_weights, vae_in, vae_out)
    feature2 = contiguous(load_file(vae_out)["feature2"])
    feature2 = contiguous(np.concatenate((np.zeros((1, 5, 128), np.float32), feature2), axis=1))
    if feature1.shape[:2] != feature2.shape[:2]:
        raise RuntimeError(f"encoder token mismatch: DINO {feature1.shape}, VAE {feature2.shape}")
    return feature1, feature2


def sample_flow(feature1, feature2, binary: Path, flow_weights: Path, work: Path,
                rng: np.random.Generator, steps: int, guidance: float):
    source, target = work / "flow-input.safetensors", work / "flow-output.safetensors"
    save_file({
        "latent": contiguous(rng.standard_normal((1, 8192, 16))),
        "camera": contiguous(rng.standard_normal((1, 1, 5))),
        "feature1": feature1,
        "feature2": feature2,
        "timestep": np.zeros((1,), np.float32),
        "positions": sobol_positions(),
        "position_freqs": position_freqs_v1(),
    }, source)
    run_worker(binary, "--run-flow", flow_weights, source, target, steps, guidance)
    tensors = load_file(target)
    return contiguous(tensors["latent"]), contiguous(tensors["camera"])


def softmax(values):
    values = values.astype(np.float32) - values.max(axis=-1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=-1, keepdims=True)


def systematic_counts(probs, counts, rng):
    output = np.zeros_like(probs, dtype=np.int64)
    for row, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        p = np.maximum(probs[row].astype(np.float64), 0)
        total = p.sum()
        p = p / total if total > 0 else np.full(len(p), 1.0 / len(p))
        cdf = np.minimum(np.cumsum(p), 1.0 - 1e-12)
        samples = rng.random() / count + np.arange(count, dtype=np.float64) / count
        indices = np.minimum(np.searchsorted(cdf, samples), len(p) - 1)
        output[row] = np.bincount(indices, minlength=len(p))
    return output


def select_points(latent, num_points: int, binary: Path, decoder_weights: Path,
                  work: Path, rng: np.random.Generator):
    coords = np.zeros((1, 3), dtype=np.int64)
    counts = np.asarray([num_points], dtype=np.int64)
    offsets = np.asarray([[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)], dtype=np.int64)
    for level in range(1, 9):
        parent_resolution = 1 << (level - 1)
        resolution = 1 << level
        source = work / f"octree-{level}-input.safetensors"
        target = work / f"octree-{level}-output.safetensors"
        save_file({
            "points": contiguous(((coords.astype(np.float32) + 0.5) / parent_resolution)[None]),
            "level": np.asarray([resolution], dtype=np.float32),
            "condition": latent,
            "position_freqs": position_freqs_v2(),
        }, source)
        run_worker(binary, "--run-octree", decoder_weights, source, target)
        logits = load_file(target)["logits"][0]
        sampled = systematic_counts(softmax(logits), counts, rng).reshape(-1)
        children = (coords[:, None, :] * 2 + offsets[None]).reshape(-1, 3)
        keep = sampled > 0
        coords, counts = children[keep], sampled[keep]
        print(f"octree level {level}: {len(coords)} occupied cells", flush=True)
    coords = np.repeat(coords, counts, axis=0)
    if len(coords) != num_points:
        raise RuntimeError(f"octree produced {len(coords)} points, expected {num_points}")
    return contiguous(((coords.astype(np.float32) + rng.random(coords.shape)) / 256.0)[None])


def radical_inverse(base: int, value: int):
    result, scale = 0.0, 1.0 / base
    while value > 0:
        result += (value % base) * scale
        value //= base
        scale /= base
    return result


def gaussian_arrays(points, features):
    anchors = points[0]
    h = features[0]
    count = len(anchors)
    per_anchor = 32
    perturb = np.asarray([[n / per_anchor, radical_inverse(2, n), radical_inverse(3, n)]
                          for n in range(per_anchor)], dtype=np.float32)
    perturb = np.arctanh((perturb * 2 - 1) / 1.5)
    inv_softplus_offset = np.log(np.expm1(np.float32(0.05)))
    offset_scale = np.logaddexp(0, h[:, 448:480].reshape(count, per_anchor, 1) + inv_softplus_offset)
    offset = h[:, 0:96].reshape(count, per_anchor, 3) + perturb
    offset = np.tanh(offset) * 0.75 * offset_scale
    xyz = (offset + anchors[:, None, :]).reshape(-1, 3) - 0.5
    dc = h[:, 96:192].reshape(-1, 3)
    scaling_raw = h[:, 192:288].reshape(-1, 3)
    rotation = h[:, 288:416].reshape(-1, 4) * 0.1
    opacity_raw = h[:, 416:448].reshape(-1, 1)
    opacity_bias = np.log(np.float32(0.1) / np.float32(0.9))
    opacity = 1.0 / (1.0 + np.exp(-(opacity_raw + opacity_bias)))
    scale_bias = np.log(np.expm1(np.float32(0.004)))
    scale = np.sqrt(np.logaddexp(0, scaling_raw + scale_bias) ** 2 + np.float32(0.0009) ** 2)
    rotation[:, 0] += 1.0
    return tuple(contiguous(x) for x in (xyz, dc, opacity, opacity_raw + opacity_bias, scale, rotation))


def quat_to_matrix(quaternion):
    q = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = q.T
    return np.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), axis=-1).reshape(-1, 3, 3)


def matrix_to_quat(matrix):
    # Stable branch implementation for proper rotation matrices.
    result = np.empty((len(matrix), 4), dtype=np.float32)
    for index, value in enumerate(matrix):
        trace = np.trace(value)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            result[index] = (0.25*s, (value[2,1]-value[1,2])/s,
                             (value[0,2]-value[2,0])/s, (value[1,0]-value[0,1])/s)
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                s = math.sqrt(1 + value[0,0] - value[1,1] - value[2,2]) * 2
                result[index] = ((value[2,1]-value[1,2])/s, 0.25*s,
                                 (value[0,1]+value[1,0])/s, (value[0,2]+value[2,0])/s)
            elif axis == 1:
                s = math.sqrt(1 + value[1,1] - value[0,0] - value[2,2]) * 2
                result[index] = ((value[0,2]-value[2,0])/s,
                                 (value[0,1]+value[1,0])/s, 0.25*s,
                                 (value[1,2]+value[2,1])/s)
            else:
                s = math.sqrt(1 + value[2,2] - value[0,0] - value[1,1]) * 2
                result[index] = ((value[1,0]-value[0,1])/s,
                                 (value[0,2]+value[2,0])/s,
                                 (value[1,2]+value[2,1])/s, 0.25*s)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def transformed(xyz, rotation):
    return contiguous(xyz @ TRANSFORM.T), contiguous(matrix_to_quat(TRANSFORM @ quat_to_matrix(rotation)))


def save_ply(path: Path, arrays):
    xyz, dc, _, opacity_logit, scale, rotation = arrays
    xyz, rotation = transformed(xyz, rotation)
    names = ("x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")
    dtype = np.dtype([(name, "<f4") for name in names])
    rows = np.empty(len(xyz), dtype=dtype)
    values = np.concatenate((xyz, np.zeros_like(xyz), dc, opacity_logit, np.log(scale), rotation), axis=1)
    rows[:] = list(map(tuple, values))
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(rows)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(rows.tobytes())


def save_splat(path: Path, arrays):
    xyz, dc, opacity, _, scale, rotation = arrays
    xyz, rotation = transformed(xyz, rotation)
    rgb = np.uint8(np.clip((dc * 0.28209479177387814 + 0.5) * 255, 0, 255))
    alpha = np.uint8(np.clip(opacity * 255, 0, 255))
    rgba = np.concatenate((rgb, alpha), axis=1)
    rotation = np.uint8(np.clip(rotation / np.linalg.norm(rotation, axis=-1, keepdims=True) * 128 + 128, 0, 255))
    order = np.argsort(-opacity[:, 0] * np.prod(scale, axis=-1))
    record = np.concatenate((
        xyz[order].astype("<f4").view(np.uint8).reshape(-1, 12),
        scale[order].astype("<f4").view(np.uint8).reshape(-1, 12),
        rgba[order], rotation[order],
    ), axis=1)
    path.write_bytes(record.reshape(-1).tobytes())


def decode(latent, num_gaussians: int, binary: Path, decoder_weights: Path,
           work: Path, rng: np.random.Generator, output_prefix: Path):
    if num_gaussians % 32:
        raise ValueError("num_gaussians must be divisible by 32")
    source = work / "decode-input.safetensors"
    save_file({"condition": latent, "position_freqs": position_freqs_v2()}, source)
    run_worker(binary, "--run-decode", decoder_weights, source, output_prefix,
               num_gaussians, int(rng.integers(0, 2**63, dtype=np.int64)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("triposplat-vulkan"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--num-gaussians", type=int, default=32768)
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.checkpoints = args.checkpoints.resolve()
    args.output = args.output.resolve()
    work = Path(tempfile.mkdtemp(prefix="triposplat-vulkan-"))
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    try:
        prepared = preprocess(args.image, args.binary,
                              args.checkpoints / "background_removal/birefnet.safetensors",
                              work, args.erode_radius)
        prepared.save(args.output.with_name(args.output.name + "-preprocessed.webp"))
        feature1, feature2 = encode(
            prepared, args.binary,
            args.checkpoints / "clip_vision/dino_v3_vit_h.safetensors",
            args.checkpoints / "vae/flux2-vae.safetensors", work, rng)
        latent, _ = sample_flow(
            feature1, feature2, args.binary,
            args.checkpoints / "diffusion_models/triposplat_fp16.safetensors",
            work, rng, args.steps, args.guidance)
        decode(latent, args.num_gaussians, args.binary,
               args.checkpoints / "vae/triposplat_vae_decoder_fp16.safetensors",
               work, rng, args.output)
        print(f"Vulkan e2e complete in {time.perf_counter() - started:.3f}s")
        print(args.output.with_suffix(".ply"))
        print(args.output.with_suffix(".splat"))
    finally:
        if args.keep_temp:
            print(f"kept temporary tensors in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
