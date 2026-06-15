#!/usr/bin/env python3
"""Build side-by-side MP4 videos for denoised |E| (after removing IMF1)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from bemd_python_loam1_pipeline import (  # noqa: E402
    center_crop,
    field_products,
    load_npz_result,
)

MAT_VAR = {"E": "a", "V1": "b", "V2": "c", "V3": "d"}


@dataclass(frozen=True)
class FrameJob:
    step: int
    left: np.ndarray
    right: np.ndarray
    left_label: str
    right_label: str


def parse_steps(start: int, end: int, stride: int) -> list[int]:
    return list(range(start, end + 1, stride))


def list_steps(results_dir: Path, steps: list[int]) -> list[int]:
    available: list[int] = []
    for step in steps:
        if (results_dir / f"step_{step:04d}.npz").exists():
            available.append(step)
    return available


def load_mat_e_residue(data_root: Path, dataset: str, step: int, mat_start_step: int) -> np.ndarray | None:
    mat_index = step - mat_start_step
    path = data_root / dataset / f"{dataset}data_BIMF{mat_index}_E.mat"
    if not path.exists():
        return None
    array = np.asarray(loadmat(path)[MAT_VAR["E"]], dtype=float)
    return array[:, :, -1]


def python_denoised(npz_path: Path) -> np.ndarray:
    components, imfs, _metadata = load_npz_result(npz_path)
    products = field_products(components, imfs)
    return products["vector_denoised_total"]


def match_shape(reference_shape: tuple[int, int], array: np.ndarray) -> np.ndarray:
    rows, cols = array.shape[:2]
    ref_rows, ref_cols = reference_shape
    if (rows, cols) == (ref_rows, ref_cols):
        return array
    if rows >= ref_rows and cols >= ref_cols:
        return center_crop(array, ref_rows if rows != cols else min(ref_rows, ref_cols))
    raise ValueError(f"Cannot match shape {array.shape} to {reference_shape}")


def compute_vmax(arrays: list[np.ndarray], percentile: float = 100.0) -> float:
    values = [float(np.nanmax(array)) for array in arrays if array.size]
    if not values:
        return 1.0
    if percentile >= 100:
        return max(values)
    return float(np.percentile(values, percentile))


def render_frame(job: FrameJob, frame_path: Path, vmax: float, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    for ax, label, array in zip(
        axes,
        (job.left_label, job.right_label),
        (job.left, job.right),
    ):
        image = ax.imshow(array, cmap="jet", origin="lower", norm=norm)
        ax.set_title(label)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, bbox_inches="tight")
    plt.close(fig)


def write_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    import cv2

    step_pattern = re.compile(r"step_(\d+)\.png$")
    frames: list[tuple[int, Path]] = []
    for path in frame_dir.glob("step_*.png"):
        match = step_pattern.search(path.name)
        if match:
            frames.append((int(match.group(1)), path))
    frames = [path for _, path in sorted(frames)]
    if not frames:
        print(f"No frames in {frame_dir}; skip video.")
        return

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read {frames[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    print(f"Video saved: {output_path}")


def build_jobs_old_new(
    dataset: str,
    steps: list[int],
    old_dir: Path,
    new_dir: Path,
) -> list[FrameJob]:
    jobs: list[FrameJob] = []
    for step in steps:
        old_path = Path(old_dir) / "results" / f"step_{step:04d}.npz"
        new_path = Path(new_dir) / "results" / f"step_{step:04d}.npz"
        if not old_path.is_file() or not new_path.is_file():
            continue
        left = python_denoised(old_path)
        right = python_denoised(new_path)
        shape = left.shape
        jobs.append(
            FrameJob(
                step=step,
                left=left,
                right=match_shape(shape, right),
                left_label="Old Python (loam*_steps)",
                right_label="New Python (vector_normal)",
            )
        )
    return jobs


def build_jobs_python_matlab(
    dataset: str,
    steps: list[int],
    python_dir: Path,
    data_root: Path,
    mat_start_step: int,
) -> list[FrameJob]:
    jobs: list[FrameJob] = []
    for step in steps:
        npz_path = Path(python_dir) / "results" / f"step_{step:04d}.npz"
        mat_residue = load_mat_e_residue(data_root, dataset, step, mat_start_step)
        if mat_residue is None or not npz_path.exists():
            continue
        py = python_denoised(npz_path)
        py = match_shape(mat_residue.shape, py)
        jobs.append(
            FrameJob(
                step=step,
                left=mat_residue,
                right=py,
                left_label="MATLAB E residue",
                right_label="Python vector denoised",
            )
        )
    return jobs


def render_dataset(
    dataset: str,
    jobs: list[FrameJob],
    output_dir: Path,
    fps: float,
    comparison_name: str,
) -> None:
    if not jobs:
        print(f"[{dataset}] no frames to render.")
        return

    vmax = compute_vmax([job.left for job in jobs] + [job.right for job in jobs])
    frame_dir = output_dir / "frames" / dataset
    for job in jobs:
        render_frame(
            job,
            frame_dir / f"step_{job.step:04d}.png",
            vmax=vmax,
            title=f"{dataset} step {job.step}: {job.left_label} | {job.right_label}",
        )

    video_path = output_dir / f"{dataset}_{comparison_name}_denoised_side_by_side.mp4"
    write_video(frame_dir, video_path, fps=fps)
    meta = {
        "dataset": dataset,
        "comparison": comparison_name,
        "steps": len(jobs),
        "vmax": vmax,
        "left_label": jobs[0].left_label,
        "right_label": jobs[0].right_label,
        "video": str(video_path),
    }
    with (output_dir / f"{dataset}_{comparison_name}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Side-by-side denoised |E| comparison MP4.")
    parser.add_argument(
        "--comparison",
        choices=("old-new", "python-matlab"),
        required=True,
    )
    parser.add_argument("--datasets", nargs="+", default=["loam0", "loam1", "loam2", "loam3"])
    parser.add_argument("--start-step", type=int, default=600)
    parser.add_argument("--end-step", type=int, default=2000)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--mat-start-step", type=int, default=600)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "python_output" / "compare_side_by_side",
    )
    parser.add_argument(
        "--old-dir-template",
        default="{dataset}_steps_0600_2000",
    )
    parser.add_argument(
        "--new-dir-template",
        default="{dataset}_vector_normal_steps_600_2000",
    )
    parser.add_argument(
        "--python-dir-template",
        default="{dataset}_vector_normal_steps_600_2000",
    )
    parser.add_argument("--data-root", type=Path, default=SCRIPT_DIR.parent / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    python_output = SCRIPT_DIR / "python_output"
    steps = parse_steps(args.start_step, args.end_step, args.step_stride)
    comparison_dir = args.output_dir / args.comparison
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        if args.comparison == "old-new":
            old_dir = python_output / args.old_dir_template.format(dataset=dataset)
            new_dir = python_output / args.new_dir_template.format(dataset=dataset)
            available = list_steps(old_dir / "results", steps)
            jobs = build_jobs_old_new(dataset, available, old_dir, new_dir)
        else:
            python_dir = python_output / args.python_dir_template.format(dataset=dataset)
            available = list_steps(python_dir / "results", steps)
            jobs = build_jobs_python_matlab(
                dataset, available, python_dir, args.data_root, args.mat_start_step
            )
        print(f"[{dataset}] rendering {len(jobs)} side-by-side frames ({args.comparison})...")
        render_dataset(dataset, jobs, comparison_dir, fps=args.fps, comparison_name=args.comparison)

    print(f"Done. Outputs in {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
