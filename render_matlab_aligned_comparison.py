#!/usr/bin/env python3
"""Render production-vs-MATLAB-aligned BEMD field comparison videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--video-seconds", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def result_files(root: Path, loam: int) -> list[Path]:
    return sorted((root / f"loam{loam}" / "results").glob("step_*.npz"))


def center_crop(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = array.shape[:2]
    out_h, out_w = shape
    top = max(0, (height - out_h) // 2)
    left = max(0, (width - out_w) // 2)
    return np.asarray(array[top : top + out_h, left : left + out_w])


def magnitude(components: list[np.ndarray]) -> np.ndarray:
    return np.sqrt(sum(np.asarray(value, dtype=np.float64) ** 2 for value in components))


def load_products(path: Path) -> dict[str, np.ndarray | int]:
    with np.load(path, allow_pickle=False) as aligned:
        metadata = json.loads(str(aligned["metadata_json"]))
        shape = tuple(int(value) for value in aligned["V1"].shape)
        raw_components = [np.asarray(aligned[name], dtype=np.float64) for name in ("V1", "V2", "V3")]
        aligned_components = [
            np.asarray(aligned[f"imf_{name}"][:, :, -1], dtype=np.float64)
            for name in ("V1", "V2", "V3")
        ]

    source_path = Path(metadata["source_npz"])
    with np.load(source_path, allow_pickle=False) as source:
        production_components = [
            center_crop(
                np.asarray(source[f"imf_{name}"][:, :, 1:], dtype=np.float64).sum(axis=-1),
                shape,
            )
            for name in ("V1", "V2", "V3")
        ]

    raw_mag = magnitude(raw_components)
    production_mag = magnitude(production_components)
    aligned_mag = magnitude(aligned_components)
    return {
        "step": int(metadata["step"]),
        "raw": raw_mag,
        "production": production_mag,
        "aligned": aligned_mag,
        "difference": np.abs(aligned_mag - production_mag),
        "production_phase": np.arctan2(production_components[1], production_components[0]),
        "aligned_phase": np.arctan2(aligned_components[1], aligned_components[0]),
        "aligned_v1": aligned_components[0],
        "aligned_v2": aligned_components[1],
    }


def panel_limits(files: list[Path]) -> dict[str, float]:
    values = {"raw": [], "production": [], "aligned": [], "difference": []}
    for path in files:
        products = load_products(path)
        for key in values:
            values[key].append(float(np.percentile(products[key], 99.5)))
    return {
        key: max(np.finfo(float).eps, float(np.median(samples)) * 1.15)
        for key, samples in values.items()
    }


def render_frame(
    products: dict[str, np.ndarray | int],
    output_path: Path,
    limits: dict[str, float],
    loam: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    image_panels = (
        ("raw", "Raw FDTD vector magnitude", "inferno"),
        ("production", "Production: remove IMF1", "inferno"),
        ("aligned", "MATLAB-aligned: remove IMF1", "inferno"),
    )
    for axis, (key, title, cmap) in zip(axes[0], image_panels):
        axis.imshow(products[key], cmap=cmap, vmin=0.0, vmax=limits[key], origin="lower")
        axis.set_title(title)

    axes[1, 0].imshow(
        products["production_phase"],
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
    )
    axes[1, 0].set_title("Production transverse phase")
    axes[1, 1].imshow(
        products["aligned_phase"],
        cmap="hsv",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
    )
    stride = max(8, int(products["aligned"].shape[0] // 24))
    rows, cols = products["aligned"].shape
    yy, xx = np.mgrid[0:rows:stride, 0:cols:stride]
    axes[1, 1].quiver(
        xx,
        yy,
        products["aligned_v1"][::stride, ::stride],
        products["aligned_v2"][::stride, ::stride],
        color="white",
        alpha=0.45,
        pivot="mid",
    )
    axes[1, 1].set_title("Aligned phase + Ex/Ey direction")
    axes[1, 2].imshow(
        products["difference"],
        cmap="magma",
        vmin=0.0,
        vmax=limits["difference"],
        origin="lower",
    )
    axes[1, 2].set_title("|aligned − production|")

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"loam {loam} · step {products['step']}", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def write_video(frames: list[Path], output_path: Path, fps: float, seconds: float) -> None:
    import imageio.v2 as imageio

    if not frames:
        return
    target_frames = max(len(frames), int(round(fps * seconds)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=float(fps),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
    ) as writer:
        for index in range(target_frames):
            source_index = min(len(frames) - 1, int(index * len(frames) / target_frames))
            writer.append_data(imageio.imread(frames[source_index]))


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    for loam in (1, 2, 3):
        files = result_files(input_dir, loam)
        if not files:
            continue
        limits = panel_limits(files)
        frames: list[Path] = []
        for index, path in enumerate(files):
            products = load_products(path)
            frame = output_dir / f"loam{loam}" / "frames" / f"frame_{index:05d}.png"
            render_frame(products, frame, limits, loam, args.dpi)
            frames.append(frame)
            print(f"[loam {loam} {index + 1}/{len(files)}] {frame}", flush=True)
        video = output_dir / f"loam{loam}" / f"matlab_aligned_loam{loam}.mp4"
        write_video(frames, video, args.fps, args.video_seconds)
        print(f"Video: {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
