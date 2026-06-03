#!/usr/bin/env python3
"""Batch BEMD pipeline for optical vortex loam1 data.

This script follows the old MATLAB-oriented optical_vortex_BEMD flow:
read one ``exy<step>.csv`` file, build E/V1/V2/V3 2D fields, run BEMD on
each component, then render per-step frames and MP4 videos for inspection.
The BEMD call is supplied by the sibling ``BEMD_Python`` project.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMPONENTS = ("E", "V1", "V2", "V3")
BEMD_COMPONENTS = ("V1", "V2", "V3")
VARIABLE_BY_COMPONENT = {"E": "a", "V1": "b", "V2": "c", "V3": "d"}
_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_BEMD_FUNC: Callable[[np.ndarray, int], np.ndarray] | None = None
_WORKER_RESULTS_DIR: Path | None = None


@dataclass
class StepMetadata:
    step: int
    source_csv: str
    raw_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    crop_size: int
    nimfs: int
    solver: str
    elapsed_seconds: float


def default_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    input_dir = script_dir.parent / "data" / "loam1"
    bemd_python_dir = repo_root / "BEMD_Python"
    return input_dir, bemd_python_dir


def parse_args() -> argparse.Namespace:
    default_input_dir, default_bemd_python_dir = default_paths()

    parser = argparse.ArgumentParser(
        description="Run BEMD_Python on optical_vortex/data/loam1 step CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--bemd-python-dir", type=Path, default=default_bemd_python_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-step", type=int, default=600)
    parser.add_argument("--end-step", type=int, default=2000)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=560)
    parser.add_argument("--nimfs", type=int, default=3)
    parser.add_argument(
        "--solver",
        choices=("direct", "gpu", "pcg"),
        default="direct",
        help="gridfit solver used by BEMD_Python.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--vector-step", type=int, default=14)
    parser.add_argument("--vector-alpha", type=float, default=1.0e-4)
    parser.add_argument(
        "--decompose-e",
        action="store_true",
        help="Also run scalar BEMD on |E|. Disabled by default; vector IMF synthesis is used for E products.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of step-level worker processes. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Thread limit exported inside each worker process.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Regenerate PNG frames and MP4 videos without recomputing existing BEMD results.",
    )
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--save-mat", action="store_true")
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for faster writes.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log failed steps and continue processing the rest.",
    )
    return parser.parse_args()


def configure_worker_thread_env(threads_per_worker: int) -> None:
    threads = str(max(1, threads_per_worker))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = threads


def configure_bemd(bemd_python_dir: Path, solver: str) -> Callable[[np.ndarray, int], np.ndarray]:
    bemd_python_dir = bemd_python_dir.resolve()
    if not bemd_python_dir.exists():
        raise FileNotFoundError(f"BEMD_Python directory not found: {bemd_python_dir}")

    sys.path.insert(0, str(bemd_python_dir))

    import bemd.sift as sift_module
    from bemd import bemd

    if solver == "gpu":
        from bemd.gridfit_gpu import gridfit_gpu

        sift_module.gridfit = gridfit_gpu
    elif solver == "pcg":
        from bemd.gridfit_pcg import gridfit_pcg

        sift_module.gridfit = gridfit_pcg

    return bemd


def step_numbers(start: int, end: int, stride: int, limit: int | None) -> Iterable[int]:
    if stride <= 0:
        raise ValueError("--step-stride must be positive.")
    count = 0
    for step in range(start, end + 1, stride):
        if limit is not None and count >= limit:
            break
        yield step
        count += 1


def center_crop(field: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0:
        return field

    rows, cols = field.shape
    out_rows = min(rows, crop_size)
    out_cols = min(cols, crop_size)
    start_row = (rows - out_rows) // 2
    start_col = (cols - out_cols) // 2
    return field[start_row : start_row + out_rows, start_col : start_col + out_cols]


def read_step_components(csv_path: Path, crop_size: int) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    data = pd.read_csv(csv_path, skiprows=3, header=None)
    if data.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {csv_path}, got {data.shape[1]}")

    x_coords = data.iloc[:, 0].to_numpy()
    y_coords = data.iloc[:, 1].to_numpy()
    ex = data.iloc[:, 2].to_numpy(dtype=np.float64)
    ey = data.iloc[:, 3].to_numpy(dtype=np.float64)
    ez = data.iloc[:, 4].to_numpy(dtype=np.float64)

    nx = len(np.unique(x_coords))
    ny = len(np.unique(y_coords))
    if nx * ny != len(data):
        raise ValueError(
            f"Grid shape mismatch for {csv_path}: nx={nx}, ny={ny}, rows={len(data)}"
        )

    # CSV rows are ordered by y with x changing fastest. For loam1 nx == ny,
    # but keeping (ny, nx) makes the axis meaning explicit.
    ex_2d = ex.reshape(ny, nx)
    ey_2d = ey.reshape(ny, nx)
    ez_2d = ez.reshape(ny, nx)
    e_abs = np.sqrt(ex_2d**2 + ey_2d**2 + ez_2d**2)

    components = {
        "E": center_crop(e_abs, crop_size),
        "V1": center_crop(ex_2d, crop_size),
        "V2": center_crop(ey_2d, crop_size),
        "V3": center_crop(ez_2d, crop_size),
    }
    return components, (ny, nx)


def run_bemd_for_step(
    bemd_func: Callable[[np.ndarray, int], np.ndarray],
    components: dict[str, np.ndarray],
    nimfs: int,
    decompose_e: bool,
) -> dict[str, np.ndarray]:
    imfs: dict[str, np.ndarray] = {}
    component_names = COMPONENTS if decompose_e else BEMD_COMPONENTS
    for name in component_names:
        print(f"    {name}: BEMD input {components[name].shape}")
        imfs[name] = bemd_func(components[name], nimfs).astype(np.float32, copy=False)
    return imfs


def save_npz(
    output_path: Path,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
    metadata: StepMetadata,
    compressed: bool,
) -> None:
    payload: dict[str, np.ndarray | str] = {
        "metadata_json": json.dumps(asdict(metadata)),
    }
    for name, array in components.items():
        payload[name] = array.astype(np.float32, copy=False)
    for name, array in imfs.items():
        payload[f"imf_{name}"] = array.astype(np.float32, copy=False)

    saver = np.savez_compressed if compressed else np.savez
    saver(output_path, **payload)


def save_mat_files(mat_dir: Path, step: int, imfs: dict[str, np.ndarray]) -> None:
    from scipy.io import savemat

    mat_dir.mkdir(parents=True, exist_ok=True)
    for component, array in sorted(imfs.items()):
        variable_name = VARIABLE_BY_COMPONENT[component]
        output_path = mat_dir / f"loam1data_step{step:04d}_BIMF0_{component}.mat"
        savemat(output_path, {variable_name: array})


def load_npz_result(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], StepMetadata]:
    with np.load(path, allow_pickle=False) as loaded:
        components = {name: loaded[name] for name in COMPONENTS}
        imfs = {
            name: loaded[f"imf_{name}"]
            for name in COMPONENTS
            if f"imf_{name}" in loaded.files
        }
        metadata = StepMetadata(**json.loads(str(loaded["metadata_json"])))
    return components, imfs, metadata


def field_products(
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    imf1_v1 = imfs["V1"][:, :, 0]
    imf1_v2 = imfs["V2"][:, :, 0]
    imf1_v3 = imfs["V3"][:, :, 0]
    imf2_v1 = imfs["V1"][:, :, 1]
    imf2_v2 = imfs["V2"][:, :, 1]
    imf2_v3 = imfs["V3"][:, :, 1]
    residue_v1 = imfs["V1"][:, :, -1]
    residue_v2 = imfs["V2"][:, :, -1]
    residue_v3 = imfs["V3"][:, :, -1]
    denoised_v1 = np.sum(imfs["V1"][:, :, 1:], axis=2)
    denoised_v2 = np.sum(imfs["V2"][:, :, 1:], axis=2)
    denoised_v3 = np.sum(imfs["V3"][:, :, 1:], axis=2)

    return {
        "original_total": components["E"],
        "imf1_total": np.sqrt(imf1_v1**2 + imf1_v2**2 + imf1_v3**2),
        "imf2_total": np.sqrt(imf2_v1**2 + imf2_v2**2 + imf2_v3**2),
        "residue_total": np.sqrt(residue_v1**2 + residue_v2**2 + residue_v3**2),
        "denoised_total": np.sqrt(denoised_v1**2 + denoised_v2**2 + denoised_v3**2),
        "imf1_v1": imf1_v1,
        "imf1_v2": imf1_v2,
        "denoised_v1": denoised_v1,
        "denoised_v2": denoised_v2,
    }


def render_summary_frame(
    frame_path: Path,
    step: int,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
    vector_step: int,
    vector_alpha: float,
) -> None:
    products = field_products(components, imfs)
    height, width = components["E"].shape
    yy, xx = np.mgrid[0:height:vector_step, 0:width:vector_step]

    fig = plt.figure(figsize=(15, 9), dpi=150)
    panels = [
        ("Original E", products["original_total"], "jet"),
        ("IMF1 magnitude", np.abs(products["imf1_total"]), "jet"),
        ("Denoised IMF2+residue", products["denoised_total"], "jet"),
    ]
    for idx, (title, array, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx)
        image = ax.imshow(array, cmap=cmap, origin="lower")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)

    vector_panels = [
        ("Original vector", components["V1"], components["V2"], products["original_total"]),
        ("IMF1 vector", products["imf1_v1"], products["imf1_v2"], np.abs(products["imf1_total"])),
        (
            "Denoised vector",
            products["denoised_v1"],
            products["denoised_v2"],
            products["denoised_total"],
        ),
    ]
    for offset, (title, v1, v2, color) in enumerate(vector_panels, start=4):
        ax = fig.add_subplot(2, 3, offset)
        ax.quiver(
            xx,
            yy,
            v1[::vector_step, ::vector_step] * vector_alpha,
            v2[::vector_step, ::vector_step] * vector_alpha,
            color[::vector_step, ::vector_step],
            cmap="Reds",
            scale=1,
            scale_units="xy",
        )
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.suptitle(f"loam1 BEMD - step {step}", fontsize=14)
    fig.tight_layout()
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, bbox_inches="tight")
    plt.close(fig)


def render_e_imf_frame(
    frame_path: Path,
    step: int,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
) -> None:
    products = field_products(components, imfs)
    labels = [
        "Original |E|",
        "|Vector IMF1|",
        "|Vector IMF2|",
        "|Vector residue|",
    ]
    arrays = [
        products["original_total"],
        products["imf1_total"],
        products["imf2_total"],
        products["residue_total"],
    ]
    cmaps = ["jet", "RdBu_r", "RdBu_r", "jet"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    for ax, label, array, cmap in zip(axes, labels, arrays, cmaps):
        image = ax.imshow(array, cmap=cmap, origin="lower")
        ax.set_title(label)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"loam1 |E| from V1/V2/V3 BEMD - step {step}", fontsize=14)
    fig.tight_layout()
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, bbox_inches="tight")
    plt.close(fig)


def render_frames(
    output_dir: Path,
    step: int,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
    vector_step: int,
    vector_alpha: float,
    overwrite: bool,
) -> None:
    summary_frame = output_dir / "frames" / "summary" / f"step_{step:04d}.png"
    e_imf_frame = output_dir / "frames" / "e_imfs" / f"step_{step:04d}.png"

    if overwrite or not summary_frame.exists():
        render_summary_frame(summary_frame, step, components, imfs, vector_step, vector_alpha)
    if overwrite or not e_imf_frame.exists():
        render_e_imf_frame(e_imf_frame, step, components, imfs)


def sorted_frame_paths(frame_dir: Path) -> list[Path]:
    step_pattern = re.compile(r"step_(\d+)\.png$")
    paths = []
    for path in frame_dir.glob("step_*.png"):
        match = step_pattern.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    return [path for _, path in sorted(paths)]


def write_video(frame_dir: Path, output_path: Path, fps: float) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for MP4 video output.") from exc

    frames = sorted_frame_paths(frame_dir)
    if not frames:
        print(f"No frames found in {frame_dir}; skipping video.")
        return

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
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
            print(f"Warning: could not read frame {frame_path}; skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    print(f"Video saved: {output_path}")


def append_error(error_log: Path, step: int, error: Exception) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"step": step, "error_type": type(error).__name__, "error": str(error)}
            )
            + "\n"
        )


def process_step(
    step: int,
    args: argparse.Namespace,
    bemd_func: Callable[[np.ndarray, int], np.ndarray],
    results_dir: Path,
) -> Path:
    csv_path = args.input_dir / f"exy{step}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing step file: {csv_path}")

    result_path = results_dir / f"step_{step:04d}.npz"
    if result_path.exists() and not args.overwrite:
        print(f"[step {step}] result exists, reusing {result_path}")
        return result_path

    print(f"[step {step}] reading {csv_path}")
    started = time.time()
    components, raw_shape = read_step_components(csv_path, args.crop_size)
    imfs = run_bemd_for_step(bemd_func, components, args.nimfs, args.decompose_e)
    elapsed = time.time() - started

    metadata = StepMetadata(
        step=step,
        source_csv=str(csv_path),
        raw_shape=raw_shape,
        cropped_shape=components["E"].shape,
        crop_size=args.crop_size,
        nimfs=args.nimfs,
        solver=args.solver,
        elapsed_seconds=elapsed,
    )
    save_npz(
        result_path,
        components,
        imfs,
        metadata,
        compressed=not args.uncompressed,
    )
    if args.save_mat:
        save_mat_files(args.output_dir / "mat", step, imfs)

    print(f"[step {step}] saved {result_path} ({elapsed:.2f}s)")
    return result_path


def init_step_worker(args: argparse.Namespace, results_dir: Path) -> None:
    global _WORKER_ARGS, _WORKER_BEMD_FUNC, _WORKER_RESULTS_DIR

    os.environ.pop("DISPLAY", None)
    configure_worker_thread_env(args.threads_per_worker)
    _WORKER_ARGS = args
    _WORKER_RESULTS_DIR = results_dir
    _WORKER_BEMD_FUNC = configure_bemd(args.bemd_python_dir, args.solver)


def process_step_in_worker(step: int) -> Path:
    if _WORKER_ARGS is None or _WORKER_BEMD_FUNC is None or _WORKER_RESULTS_DIR is None:
        raise RuntimeError("Worker was not initialized.")
    return process_step(step, _WORKER_ARGS, _WORKER_BEMD_FUNC, _WORKER_RESULTS_DIR)


def render_result_if_needed(args: argparse.Namespace, result_path: Path) -> None:
    if args.no_frames:
        return

    components, imfs, metadata = load_npz_result(result_path)
    render_frames(
        args.output_dir,
        metadata.step,
        components,
        imfs,
        args.vector_step,
        args.vector_alpha,
        overwrite=args.overwrite or args.overwrite_frames,
    )


def run_steps_serial(
    args: argparse.Namespace,
    steps: list[int],
    bemd_func: Callable[[np.ndarray, int], np.ndarray],
    results_dir: Path,
    error_log: Path,
) -> list[Path]:
    processed_results: list[Path] = []
    for step in steps:
        try:
            result_path = process_step(step, args, bemd_func, results_dir)
            processed_results.append(result_path)
            render_result_if_needed(args, result_path)
        except Exception as exc:
            append_error(error_log, step, exc)
            print(f"[step {step}] ERROR: {exc}")
            if not args.continue_on_error:
                raise
    return processed_results


def run_steps_parallel(
    args: argparse.Namespace,
    steps: list[int],
    results_dir: Path,
    error_log: Path,
) -> list[Path]:
    processed_results: list[Path] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_step_worker,
        initargs=(args, results_dir),
    ) as executor:
        future_by_step = {executor.submit(process_step_in_worker, step): step for step in steps}
        for future in as_completed(future_by_step):
            step = future_by_step[future]
            try:
                result_path = future.result()
                processed_results.append(result_path)
                render_result_if_needed(args, result_path)
            except Exception as exc:
                append_error(error_log, step, exc)
                print(f"[step {step}] ERROR: {exc}")
                if not args.continue_on_error:
                    for pending in future_by_step:
                        pending.cancel()
                    raise
    return processed_results


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    args.input_dir = args.input_dir.resolve()
    args.bemd_python_dir = args.bemd_python_dir.resolve()
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "python_output"
            / f"loam1_steps_{args.start_step:04d}_{args.end_step:04d}"
        )
    args.output_dir = args.output_dir.resolve()

    os.environ.pop("DISPLAY", None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: str(value) for key, value in vars(args).items()}, handle, indent=2)

    missing_steps: list[int] = []
    error_log = args.output_dir / "errors.jsonl"

    print("=== loam1 BEMD Python batch ===")
    print(f"input:  {args.input_dir}")
    print(f"output: {args.output_dir}")
    print(f"steps:  {args.start_step}..{args.end_step} stride={args.step_stride}")
    print(f"crop:   {args.crop_size}, nimfs={args.nimfs}, solver={args.solver}")
    print(f"bemd components: {', '.join(COMPONENTS if args.decompose_e else BEMD_COMPONENTS)}")
    print(f"workers: {args.workers}, threads/worker: {args.threads_per_worker}")

    steps_to_process: list[int] = []
    for step in step_numbers(args.start_step, args.end_step, args.step_stride, args.limit_steps):
        csv_path = args.input_dir / f"exy{step}.csv"
        if not csv_path.exists():
            missing_steps.append(step)
            print(f"[step {step}] missing {csv_path.name}, skipping.")
            continue
        steps_to_process.append(step)

    if args.workers == 1:
        configure_worker_thread_env(args.threads_per_worker)
        bemd_func = configure_bemd(args.bemd_python_dir, args.solver)
        processed_results = run_steps_serial(args, steps_to_process, bemd_func, results_dir, error_log)
    else:
        processed_results = run_steps_parallel(args, steps_to_process, results_dir, error_log)

    if missing_steps:
        with (args.output_dir / "missing_steps.json").open("w", encoding="utf-8") as handle:
            json.dump(missing_steps, handle, indent=2)
        print(f"Skipped {len(missing_steps)} missing step files.")

    if not args.no_video and not args.no_frames:
        write_video(
            args.output_dir / "frames" / "summary",
            args.output_dir / f"loam1_bemd_{args.start_step:04d}_{args.end_step:04d}_summary.mp4",
            args.fps,
        )
        write_video(
            args.output_dir / "frames" / "e_imfs",
            args.output_dir / f"loam1_bemd_{args.start_step:04d}_{args.end_step:04d}_E_imfs.mp4",
            args.fps,
        )

    print(f"Processed or reused {len(processed_results)} result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
