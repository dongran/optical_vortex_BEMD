#!/usr/bin/env python3
"""Compare legacy MATLAB BEMD .mat files against Python pipeline .npz results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from bemd_python_loam1_pipeline import field_products, load_npz_result


MAT_VARIABLE_BY_COMPONENT = {"E": "a", "V1": "b", "V2": "c", "V3": "d"}


def parse_steps(value: str) -> list[int]:
    steps: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            chunks = [int(chunk) for chunk in part.split(":")]
            if len(chunks) == 2:
                start, end = chunks
                stride = 1
            elif len(chunks) == 3:
                start, end, stride = chunks
            else:
                raise ValueError(f"Invalid step range: {part}")
            steps.extend(range(start, end + 1, stride))
        else:
            steps.append(int(part))
    return sorted(dict.fromkeys(steps))


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    parser = argparse.ArgumentParser(
        description="Compare MATLAB legacy optical-vortex BEMD .mat files to Python .npz results."
    )
    parser.add_argument("--datasets", nargs="+", default=["loam1", "loam3"])
    parser.add_argument("--steps", default="600,800,1000,1400,1800,1999")
    parser.add_argument("--mat-start-step", type=int, default=600)
    parser.add_argument("--data-root", type=Path, default=script_dir.parent / "data")
    parser.add_argument("--python-output-root", type=Path, default=script_dir / "python_output")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "python_output" / "matlab_python_compare")
    parser.add_argument(
        "--python-dir-template",
        default="{dataset}_steps_0600_2000",
        help="Directory under --python-output-root containing results/step_XXXX.npz.",
    )
    parser.add_argument(
        "--comparison-name",
        default="current",
        help="Subdirectory name for this comparison run.",
    )
    parser.add_argument("--make-video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def center_crop_to_shape(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = array.shape[:2]
    out_rows, out_cols = shape
    if rows == out_rows and cols == out_cols:
        return array
    if rows < out_rows or cols < out_cols:
        raise ValueError(f"Cannot crop {array.shape} to {shape}")
    start_row = (rows - out_rows) // 2
    start_col = (cols - out_cols) // 2
    return array[start_row : start_row + out_rows, start_col : start_col + out_cols]


def finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_flat = np.asarray(a, dtype=float).ravel()
    b_flat = np.asarray(b, dtype=float).ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    return a_flat[mask], b_flat[mask]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = finite_pair(a, b)
    if a_flat.size < 2 or np.std(a_flat) == 0 or np.std(b_flat) == 0:
        return float("nan")
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def rel_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref, cand = finite_pair(reference, candidate)
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(cand - ref) / denom) if denom else float("nan")


def nrmse(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref, cand = finite_pair(reference, candidate)
    if ref.size == 0:
        return float("nan")
    rmse = np.sqrt(np.mean((cand - ref) ** 2))
    spread = np.max(ref) - np.min(ref)
    return float(rmse / spread) if spread else float("nan")


def mat_path(data_root: Path, dataset: str, mat_index: int, component: str) -> Path:
    return data_root / dataset / f"{dataset}data_BIMF{mat_index}_{component}.mat"


def load_mat_component(data_root: Path, dataset: str, mat_index: int, component: str) -> np.ndarray | None:
    path = mat_path(data_root, dataset, mat_index, component)
    if not path.exists():
        return None
    payload = loadmat(path)
    return np.asarray(payload[MAT_VARIABLE_BY_COMPONENT[component]], dtype=float)


def python_result_path(args: argparse.Namespace, dataset: str, step: int) -> Path:
    dirname = args.python_dir_template.format(dataset=dataset)
    return args.python_output_root / dirname / "results" / f"step_{step:04d}.npz"


def add_metric(
    rows: list[dict[str, object]],
    dataset: str,
    step: int,
    mat_index: int,
    reference_name: str,
    candidate_name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> None:
    rows.append(
        {
            "dataset": dataset,
            "step": step,
            "mat_index": mat_index,
            "reference": reference_name,
            "candidate": candidate_name,
            "pearson": pearson(reference, candidate),
            "rel_l2": rel_l2(reference, candidate),
            "nrmse": nrmse(reference, candidate),
        }
    )


def plot_comparison(
    output_path: Path,
    dataset: str,
    step: int,
    mat_e_residue: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> None:
    panels: list[tuple[str, np.ndarray, str]] = [("MATLAB E residue", mat_e_residue, "jet")]
    for name, array in candidates.items():
        panels.append((name, array, "jet"))
        panels.append((f"diff: {name}", array - mat_e_residue, "RdBu_r"))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), dpi=140)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (title, array, cmap) in zip(axes_array, panels):
        if cmap == "RdBu_r":
            vmax = float(np.nanmax(np.abs(array))) or 1.0
            image = ax.imshow(array, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        else:
            image = ax.imshow(array, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)
    for ax in axes_array[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(f"{dataset} step {step}: MATLAB vs Python BEMD", fontsize=14)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sorted_pngs(frame_dir: Path) -> list[Path]:
    return sorted(frame_dir.glob("*.png"))


def write_video(frame_dir: Path, output_path: Path, fps: float = 2.0) -> None:
    frames = sorted_pngs(frame_dir)
    if not frames:
        return
    try:
        import cv2
    except ImportError:
        print("opencv-python unavailable; skipping comparison video.")
        return

    first = cv2.imread(str(frames[0]))
    if first is None:
        return
    height, width = first.shape[:2]
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


def compare_dataset_step(
    args: argparse.Namespace,
    dataset: str,
    step: int,
    rows: list[dict[str, object]],
) -> Path | None:
    mat_index = step - args.mat_start_step
    if mat_index < 0:
        return None

    mat_e = load_mat_component(args.data_root, dataset, mat_index, "E")
    npz_path = python_result_path(args, dataset, step)
    if mat_e is None or not npz_path.exists():
        print(f"[{dataset} step {step}] missing MATLAB E or Python npz; skipping.")
        return None

    components, imfs, _metadata = load_npz_result(npz_path)
    products = field_products(components, imfs)
    mat_shape = mat_e.shape[:2]
    mat_e_residue = mat_e[:, :, -1]

    candidates = {
        "Python vector denoised": center_crop_to_shape(products["vector_denoised_total"], mat_shape),
        "Python vector residue": center_crop_to_shape(products["vector_residue_total"], mat_shape),
    }
    if "e_residue" in products:
        candidates["Python scalar E residue"] = center_crop_to_shape(products["e_residue"], mat_shape)

    for name, candidate in candidates.items():
        add_metric(rows, dataset, step, mat_index, "MATLAB E residue", name, mat_e_residue, candidate)

    mat_vectors = []
    for component in ("V1", "V2", "V3"):
        mat_component = load_mat_component(args.data_root, dataset, mat_index, component)
        if mat_component is None:
            continue
        mat_vectors.append(mat_component[:, :, -1])
        py_residue = center_crop_to_shape(products[f"residue_{component.lower()}"], mat_shape)
        py_denoised = center_crop_to_shape(products[f"denoised_{component.lower()}"], mat_shape)
        add_metric(
            rows,
            dataset,
            step,
            mat_index,
            f"MATLAB {component} residue",
            f"Python {component} residue",
            mat_component[:, :, -1],
            py_residue,
        )
        add_metric(
            rows,
            dataset,
            step,
            mat_index,
            f"MATLAB {component} residue",
            f"Python {component} denoised",
            mat_component[:, :, -1],
            py_denoised,
        )

    if len(mat_vectors) == 3:
        mat_vector_total = np.sqrt(sum(vector**2 for vector in mat_vectors))
        add_metric(
            rows,
            dataset,
            step,
            mat_index,
            "MATLAB vector residue magnitude",
            "Python vector residue",
            mat_vector_total,
            candidates["Python vector residue"],
        )
        add_metric(
            rows,
            dataset,
            step,
            mat_index,
            "MATLAB vector residue magnitude",
            "Python vector denoised",
            mat_vector_total,
            candidates["Python vector denoised"],
        )

    frame_path = (
        args.output_dir
        / args.comparison_name
        / "frames"
        / dataset
        / f"{dataset}_step_{step:04d}_compare.png"
    )
    plot_comparison(frame_path, dataset, step, mat_e_residue, candidates)
    return frame_path


def print_summary(rows: Iterable[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["reference"]), str(row["candidate"]))
        value = float(row["pearson"])
        if np.isfinite(value):
            grouped.setdefault(key, []).append(value)
    for key, values in sorted(grouped.items()):
        dataset, reference, candidate = key
        print(
            f"{dataset}: {reference} vs {candidate}: "
            f"mean corr={np.mean(values):.6f}, min corr={np.min(values):.6f}, n={len(values)}"
        )


def main() -> int:
    args = parse_args()
    args.steps = parse_steps(args.steps)
    args.data_root = args.data_root.resolve()
    args.python_output_root = args.python_output_root.resolve()
    args.output_dir = args.output_dir.resolve()

    rows: list[dict[str, object]] = []
    frame_paths: list[Path] = []
    for dataset in args.datasets:
        for step in args.steps:
            frame_path = compare_dataset_step(args, dataset, step, rows)
            if frame_path is not None:
                frame_paths.append(frame_path)

    run_dir = args.output_dir / args.comparison_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "comparison_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    write_csv(run_dir / "comparison_metrics.csv", rows)

    if args.make_video:
        for dataset in args.datasets:
            write_video(run_dir / "frames" / dataset, run_dir / f"{dataset}_comparison.mp4")

    print(f"Wrote {len(rows)} metrics and {len(frame_paths)} frames to {run_dir}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
