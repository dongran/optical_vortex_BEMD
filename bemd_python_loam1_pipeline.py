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
import matplotlib.colors as mcolors
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


@dataclass(frozen=True)
class ColorLimits:
    """Fixed imshow color ranges for an entire step batch / video."""

    summary_original_vmax: float
    summary_imf1_mag_vmax: float
    summary_denoised_vmax: float
    e_imf_original_vmax: float
    e_imf_imf1_sym: float
    e_imf_imf2_sym: float
    e_imf_denoised_vmax: float
    e_imf_residue_vmax: float
    color_scale: str = "max"
    color_percentile: float = 100.0

    @property
    def summary_quiver_vmax(self) -> float:
        return self.summary_original_vmax

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, float | str]) -> ColorLimits:
        return cls(
            summary_original_vmax=float(payload["summary_original_vmax"]),
            summary_imf1_mag_vmax=float(payload["summary_imf1_mag_vmax"]),
            summary_denoised_vmax=float(payload["summary_denoised_vmax"]),
            e_imf_original_vmax=float(payload["e_imf_original_vmax"]),
            e_imf_imf1_sym=float(payload["e_imf_imf1_sym"]),
            e_imf_imf2_sym=float(payload["e_imf_imf2_sym"]),
            e_imf_denoised_vmax=float(
                payload.get("e_imf_denoised_vmax", payload["summary_denoised_vmax"])
            ),
            e_imf_residue_vmax=float(payload["e_imf_residue_vmax"]),
            color_scale=str(payload.get("color_scale", "max")),
            color_percentile=float(payload.get("color_percentile", 100.0)),
        )


def infer_dataset_label(input_dir: Path, output_dir: Path | None = None) -> str:
    """Derive loam1/loam3-style label from input or output directory name."""
    input_name = input_dir.name.lower()
    if input_name.startswith("loam"):
        return input_name

    if output_dir is not None:
        match = re.match(r"^(loam\d+)_steps_", output_dir.name, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    return input_name or "dataset"


def default_paths() -> tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    input_dir = script_dir.parent / "data" / "loam1"
    bemd_python_dir = repo_root / "BEMD_Python"
    return input_dir, bemd_python_dir


def parse_args() -> argparse.Namespace:
    default_input_dir, default_bemd_python_dir = default_paths()

    parser = argparse.ArgumentParser(
        description="Run BEMD_Python on optical_vortex/data/loam* step CSV files."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument(
        "--dataset-label",
        type=str,
        default=None,
        help="Prefix for output videos and plots (default: inferred from --input-dir, e.g. loam3).",
    )
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
        "--reconstruction-mode",
        choices=("vector-denoised", "matlab-e-residue", "matlab-vector-residue"),
        default="vector-denoised",
        help=(
            "Field used for the denoised summary panel: current vector IMF2+residue "
            "default, MATLAB-like scalar E residue, or MATLAB-like vector residue."
        ),
    )
    parser.add_argument(
        "--sift-cost-threshold",
        type=float,
        default=0.2,
        help="BEMD sift SD stopping threshold; MATLAB legacy default is 0.2.",
    )
    parser.add_argument(
        "--sift-max-iterations",
        type=int,
        default=100,
        help="Maximum sift iterations. Use 0 to mimic MATLAB's unbounded while loop.",
    )
    parser.add_argument(
        "--gridfit-smoothness",
        type=float,
        default=1.0,
        help="Smoothness forwarded to BEMD_Python gridfit; MATLAB default is 1.0.",
    )
    parser.add_argument(
        "--gridfit-linear-solver",
        choices=("normal", "backslash"),
        default="normal",
        help=(
            "Linear solver inside BEMD_Python gridfit. 'normal' is the historical "
            "Python path; 'backslash' uses LSQR on the augmented system to better "
            "match MATLAB gridfit's default A\\rhs behavior."
        ),
    )
    parser.add_argument(
        "--matlab-compatible",
        action="store_true",
        help=(
            "Strict bemd.m compatibility: unbounded sift, augmented backslash/LSQR, "
            "and errors instead of Python's insufficient-extrema/NaN fallbacks."
        ),
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
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip BEMD; recompute color limits, regenerate PNG frames and MP4 from existing .npz.",
    )
    parser.add_argument(
        "--color-scale",
        choices=("max", "mean", "percentile"),
        default="max",
        help="How to set fixed colorbar vmax across all steps (default: global max).",
    )
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=99.5,
        help="Used when --color-scale=percentile (0-100, applied per step then aggregated).",
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


def configure_bemd(args: argparse.Namespace) -> Callable[[np.ndarray, int], np.ndarray]:
    bemd_python_dir = args.bemd_python_dir
    solver = args.solver
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

    max_iterations = args.sift_max_iterations if args.sift_max_iterations > 0 else None

    def configured_bemd(input_image: np.ndarray, nimfs: int) -> np.ndarray:
        return bemd(
            input_image,
            nimfs,
            cost_threshold=args.sift_cost_threshold,
            max_iterations=max_iterations,
            gridfit_smoothness=args.gridfit_smoothness,
            gridfit_solver=args.gridfit_linear_solver,
            matlab_compatible=bool(args.matlab_compatible),
        )

    return configured_bemd


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

    # Build the physical (y, x) grid from coordinates instead of relying on CSV
    # row order. This removes the square-grid reshape/transpose ambiguity between
    # the legacy MATLAB loader and the original Python port.
    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    x_index = np.searchsorted(x_unique, x_coords)
    y_index = np.searchsorted(y_unique, y_coords)
    if (
        np.any(x_index < 0)
        or np.any(x_index >= nx)
        or np.any(y_index < 0)
        or np.any(y_index >= ny)
    ):
        raise ValueError(f"Coordinate indexing failed for {csv_path}")
    flat_index = y_index * nx + x_index
    if np.unique(flat_index).size != flat_index.size:
        raise ValueError(f"Duplicate (x,y) cells found in {csv_path}")
    occupied = np.zeros((ny, nx), dtype=bool)
    occupied[y_index, x_index] = True
    if not np.all(occupied):
        raise ValueError(f"Missing (x,y) cells found in {csv_path}")
    ex_2d = np.empty((ny, nx), dtype=np.float64)
    ey_2d = np.empty((ny, nx), dtype=np.float64)
    ez_2d = np.empty((ny, nx), dtype=np.float64)
    ex_2d[y_index, x_index] = ex
    ey_2d[y_index, x_index] = ey
    ez_2d[y_index, x_index] = ez
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


def save_mat_files(
    mat_dir: Path,
    step: int,
    imfs: dict[str, np.ndarray],
    dataset_label: str,
) -> None:
    from scipy.io import savemat

    mat_dir.mkdir(parents=True, exist_ok=True)
    for component, array in sorted(imfs.items()):
        variable_name = VARIABLE_BY_COMPONENT[component]
        output_path = mat_dir / f"{dataset_label}data_step{step:04d}_BIMF0_{component}.mat"
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
    def layer(name: str, index: int) -> np.ndarray:
        array = imfs[name]
        if index < array.shape[2]:
            return array[:, :, index]
        return np.zeros_like(array[:, :, 0])

    imf1_v1 = imfs["V1"][:, :, 0]
    imf1_v2 = imfs["V2"][:, :, 0]
    imf1_v3 = imfs["V3"][:, :, 0]
    imf2_v1 = layer("V1", 1) if imfs["V1"].shape[2] > 2 else np.zeros_like(imf1_v1)
    imf2_v2 = layer("V2", 1) if imfs["V2"].shape[2] > 2 else np.zeros_like(imf1_v2)
    imf2_v3 = layer("V3", 1) if imfs["V3"].shape[2] > 2 else np.zeros_like(imf1_v3)
    residue_v1 = imfs["V1"][:, :, -1]
    residue_v2 = imfs["V2"][:, :, -1]
    residue_v3 = imfs["V3"][:, :, -1]
    denoised_v1 = np.sum(imfs["V1"][:, :, 1:], axis=2)
    denoised_v2 = np.sum(imfs["V2"][:, :, 1:], axis=2)
    denoised_v3 = np.sum(imfs["V3"][:, :, 1:], axis=2)

    products = {
        "original_total": components["E"],
        "vector_imf1_total": np.sqrt(imf1_v1**2 + imf1_v2**2 + imf1_v3**2),
        "vector_imf2_total": np.sqrt(imf2_v1**2 + imf2_v2**2 + imf2_v3**2),
        "vector_residue_total": np.sqrt(residue_v1**2 + residue_v2**2 + residue_v3**2),
        "vector_denoised_total": np.sqrt(denoised_v1**2 + denoised_v2**2 + denoised_v3**2),
        "imf1_v1": imf1_v1,
        "imf1_v2": imf1_v2,
        "denoised_v1": denoised_v1,
        "denoised_v2": denoised_v2,
        "denoised_v3": denoised_v3,
        "residue_v1": residue_v1,
        "residue_v2": residue_v2,
        "residue_v3": residue_v3,
    }
    # Backward-compatible aliases used by the default visualization.
    products["imf1_total"] = products["vector_imf1_total"]
    products["imf2_total"] = products["vector_imf2_total"]
    products["residue_total"] = products["vector_residue_total"]
    products["denoised_total"] = products["vector_denoised_total"]

    if "E" in imfs:
        e_imf1 = imfs["E"][:, :, 0]
        e_imf2 = imfs["E"][:, :, 1] if imfs["E"].shape[2] > 2 else np.zeros_like(e_imf1)
        e_residue = imfs["E"][:, :, -1]
        products.update(
            {
                "e_imf1": e_imf1,
                "e_imf2": e_imf2,
                "e_residue": e_residue,
                "e_denoised": np.sum(imfs["E"][:, :, 1:], axis=2),
            }
        )
    return products


def select_reconstruction(
    products: dict[str, np.ndarray],
    mode: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "vector-denoised":
        return (
            "Vector |E| without IMF1",
            products["vector_denoised_total"],
            products["denoised_v1"],
            products["denoised_v2"],
        )
    if mode == "matlab-vector-residue":
        return (
            "MATLAB-like vector residue",
            products["vector_residue_total"],
            products["residue_v1"],
            products["residue_v2"],
        )
    if "e_residue" not in products:
        raise ValueError(
            "--reconstruction-mode matlab-e-residue requires results generated with --decompose-e."
        )
    return (
        "Scalar E residue (MATLAB diagnostic)",
        products["e_residue"],
        products["residue_v1"],
        products["residue_v2"],
    )


def e_imf_panel_products(
    products: dict[str, np.ndarray],
    mode: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    if mode == "matlab-e-residue":
        if "e_residue" not in products:
            raise ValueError(
                "--reconstruction-mode matlab-e-residue requires results generated with --decompose-e."
            )
        return (
            "Scalar E",
            products["e_imf1"],
            products["e_denoised"],
            products["e_residue"],
        )
    return (
        "Vector magnitude",
        products["vector_imf1_total"],
        products["vector_denoised_total"],
        products["vector_residue_total"],
    )


def _step_scalar_stats(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(np.min(finite)), float(np.max(finite))


def _step_symmetric_stat(array: np.ndarray) -> float:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def _aggregate_step_values(
    step_values: list[float],
    color_scale: str,
    color_percentile: float,
) -> float:
    if not step_values:
        return 1.0
    if color_scale == "max":
        return max(step_values)
    if color_scale == "mean":
        return float(np.mean(step_values))
    return float(np.percentile(step_values, color_percentile))


def compute_color_limits(
    result_paths: Iterable[Path],
    color_scale: str,
    color_percentile: float,
    reconstruction_mode: str,
) -> ColorLimits:
    summary_original: list[float] = []
    summary_imf1_mag: list[float] = []
    summary_denoised: list[float] = []
    e_imf_original: list[float] = []
    e_imf_imf1_sym: list[float] = []
    e_imf_imf2_sym: list[float] = []
    e_imf_denoised: list[float] = []
    e_imf_residue: list[float] = []

    for result_path in result_paths:
        components, imfs, _metadata = load_npz_result(result_path)
        products = field_products(components, imfs)
        _selected_label, selected_total, _selected_v1, _selected_v2 = select_reconstruction(
            products, reconstruction_mode
        )
        _panel_prefix, panel_imf1, panel_denoised, panel_residue = e_imf_panel_products(
            products, reconstruction_mode
        )
        summary_original.append(_step_scalar_stats(products["original_total"])[1])
        summary_imf1_mag.append(_step_scalar_stats(np.abs(products["vector_imf1_total"]))[1])
        summary_denoised.append(_step_scalar_stats(selected_total)[1])
        e_imf_original.append(_step_scalar_stats(products["original_total"])[1])
        e_imf_imf1_sym.append(_step_symmetric_stat(panel_imf1))
        e_imf_imf2_sym.append(_step_symmetric_stat(products["vector_imf2_total"]))
        e_imf_denoised.append(_step_scalar_stats(panel_denoised)[1])
        e_imf_residue.append(_step_scalar_stats(panel_residue)[1])

    def vmax(values: list[float]) -> float:
        value = _aggregate_step_values(values, color_scale, color_percentile)
        return value if value > 0 else 1.0

    return ColorLimits(
        summary_original_vmax=vmax(summary_original),
        summary_imf1_mag_vmax=vmax(summary_imf1_mag),
        summary_denoised_vmax=vmax(summary_denoised),
        e_imf_original_vmax=vmax(e_imf_original),
        e_imf_imf1_sym=vmax(e_imf_imf1_sym),
        e_imf_imf2_sym=vmax(e_imf_imf2_sym),
        e_imf_denoised_vmax=vmax(e_imf_denoised),
        e_imf_residue_vmax=vmax(e_imf_residue),
        color_scale=color_scale,
        color_percentile=color_percentile,
    )


def color_limits_path(output_dir: Path) -> Path:
    return output_dir / "color_limits.json"


def load_color_limits(path: Path) -> ColorLimits:
    with path.open(encoding="utf-8") as handle:
        return ColorLimits.from_dict(json.load(handle))


def save_color_limits(output_dir: Path, limits: ColorLimits) -> Path:
    path = color_limits_path(output_dir)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(limits.to_dict(), handle, indent=2)
    return path


def mode_suffix(reconstruction_mode: str) -> str:
    return "" if reconstruction_mode == "vector-denoised" else f"_{reconstruction_mode}"


def frame_dir(output_dir: Path, kind: str, reconstruction_mode: str) -> Path:
    if reconstruction_mode == "vector-denoised":
        return output_dir / "frames" / kind
    return output_dir / "frames" / reconstruction_mode / kind


def video_path(
    output_dir: Path,
    dataset_label: str,
    start_step: int,
    end_step: int,
    kind: str,
    reconstruction_mode: str,
) -> Path:
    suffix = mode_suffix(reconstruction_mode)
    return output_dir / f"{dataset_label}_bemd_{start_step:04d}_{end_step:04d}{suffix}_{kind}.mp4"


def list_result_paths(
    results_dir: Path,
    allowed_steps: Iterable[int] | None = None,
) -> list[Path]:
    step_pattern = re.compile(r"step_(\d+)\.npz$")
    allowed = set(allowed_steps) if allowed_steps is not None else None
    paths: list[tuple[int, Path]] = []
    for path in results_dir.glob("step_*.npz"):
        match = step_pattern.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if allowed is not None and step not in allowed:
            continue
        paths.append((step, path))
    return [path for _, path in sorted(paths)]


def render_summary_frame(
    frame_path: Path,
    step: int,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
    vector_step: int,
    vector_alpha: float,
    dataset_label: str,
    color_limits: ColorLimits,
    reconstruction_mode: str,
) -> None:
    products = field_products(components, imfs)
    selected_label, selected_total, selected_v1, selected_v2 = select_reconstruction(
        products, reconstruction_mode
    )
    height, width = components["E"].shape
    yy, xx = np.mgrid[0:height:vector_step, 0:width:vector_step]

    fig = plt.figure(figsize=(15, 9), dpi=150)
    panels = [
        ("Original E", products["original_total"], "jet", color_limits.summary_original_vmax),
        (
            "IMF1 magnitude",
            np.abs(products["imf1_total"]),
            "jet",
            color_limits.summary_imf1_mag_vmax,
        ),
        (
            selected_label,
            selected_total,
            "jet",
            color_limits.summary_denoised_vmax,
        ),
    ]
    for idx, (title, array, cmap, vmax) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, idx)
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        image = ax.imshow(array, cmap=cmap, origin="lower", norm=norm)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)

    quiver_norm = mcolors.Normalize(vmin=0.0, vmax=color_limits.summary_quiver_vmax)
    vector_panels = [
        ("Original vector", components["V1"], components["V2"], products["original_total"]),
        ("IMF1 vector", products["imf1_v1"], products["imf1_v2"], np.abs(products["imf1_total"])),
        (
            selected_label,
            selected_v1,
            selected_v2,
            selected_total,
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
            norm=quiver_norm,
            scale=1,
            scale_units="xy",
        )
        ax.set_title(title)
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.set_axis_off()

    fig.suptitle(f"{dataset_label} BEMD ({reconstruction_mode}) - step {step}", fontsize=14)
    fig.tight_layout()
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(frame_path, bbox_inches="tight")
    plt.close(fig)


def render_e_imf_frame(
    frame_path: Path,
    step: int,
    components: dict[str, np.ndarray],
    imfs: dict[str, np.ndarray],
    dataset_label: str,
    color_limits: ColorLimits,
    reconstruction_mode: str,
) -> None:
    products = field_products(components, imfs)
    panel_prefix, panel_imf1, panel_denoised, panel_residue = e_imf_panel_products(
        products, reconstruction_mode
    )
    panels = [
        ("Original |E|", products["original_total"], "jet", 0.0, color_limits.e_imf_original_vmax),
        (
            f"{panel_prefix} IMF1",
            panel_imf1,
            "RdBu_r",
            -color_limits.e_imf_imf1_sym,
            color_limits.e_imf_imf1_sym,
        ),
        (
            f"{panel_prefix} without IMF1",
            panel_denoised,
            "jet",
            0.0,
            color_limits.e_imf_denoised_vmax,
        ),
        (
            f"{panel_prefix} residue only",
            panel_residue,
            "jet",
            0.0,
            color_limits.e_imf_residue_vmax,
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
    for ax, (label, array, cmap, vmin, vmax) in zip(axes, panels):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        image = ax.imshow(array, cmap=cmap, origin="lower", norm=norm)
        ax.set_title(label)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(f"{dataset_label} BEMD layers ({reconstruction_mode}) - step {step}", fontsize=14)
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
    dataset_label: str,
    color_limits: ColorLimits,
    reconstruction_mode: str,
) -> None:
    summary_frame = frame_dir(output_dir, "summary", reconstruction_mode) / f"step_{step:04d}.png"
    e_imf_frame = frame_dir(output_dir, "e_imfs", reconstruction_mode) / f"step_{step:04d}.png"

    if overwrite or not summary_frame.exists():
        render_summary_frame(
            summary_frame,
            step,
            components,
            imfs,
            vector_step,
            vector_alpha,
            dataset_label,
            color_limits,
            reconstruction_mode,
        )
    if overwrite or not e_imf_frame.exists():
        render_e_imf_frame(
            e_imf_frame,
            step,
            components,
            imfs,
            dataset_label,
            color_limits,
            reconstruction_mode,
        )


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
        save_mat_files(args.output_dir / "mat", step, imfs, args.dataset_label)

    print(f"[step {step}] saved {result_path} ({elapsed:.2f}s)")
    return result_path


def init_step_worker(args: argparse.Namespace, results_dir: Path) -> None:
    global _WORKER_ARGS, _WORKER_BEMD_FUNC, _WORKER_RESULTS_DIR

    os.environ.pop("DISPLAY", None)
    configure_worker_thread_env(args.threads_per_worker)
    _WORKER_ARGS = args
    _WORKER_RESULTS_DIR = results_dir
    _WORKER_BEMD_FUNC = configure_bemd(args)


def process_step_in_worker(step: int) -> Path:
    if _WORKER_ARGS is None or _WORKER_BEMD_FUNC is None or _WORKER_RESULTS_DIR is None:
        raise RuntimeError("Worker was not initialized.")
    return process_step(step, _WORKER_ARGS, _WORKER_BEMD_FUNC, _WORKER_RESULTS_DIR)


def render_result_frame(
    args: argparse.Namespace,
    result_path: Path,
    color_limits: ColorLimits,
    overwrite: bool,
) -> None:
    components, imfs, metadata = load_npz_result(result_path)
    render_frames(
        args.output_dir,
        metadata.step,
        components,
        imfs,
        args.vector_step,
        args.vector_alpha,
        overwrite=overwrite,
        dataset_label=args.dataset_label,
        color_limits=color_limits,
        reconstruction_mode=args.reconstruction_mode,
    )


def render_all_frames(
    args: argparse.Namespace,
    result_paths: list[Path],
    color_limits: ColorLimits,
) -> None:
    if args.no_frames or not result_paths:
        return

    overwrite = args.overwrite or args.overwrite_frames or args.render_only
    total = len(result_paths)
    print(
        f"Rendering {total} frame pairs with fixed color limits "
        f"({args.color_scale}, {args.reconstruction_mode})..."
    )

    if args.workers <= 1:
        for index, result_path in enumerate(result_paths, start=1):
            render_result_frame(args, result_path, color_limits, overwrite=overwrite)
            if index % 100 == 0 or index == total:
                print(f"  rendered {index}/{total}")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(render_result_frame, args, path, color_limits, overwrite): path
            for path in result_paths
        }
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  rendered {done}/{total}")


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
    if args.dataset_label is None:
        args.dataset_label = infer_dataset_label(args.input_dir, args.output_dir)
    else:
        args.dataset_label = args.dataset_label.lower()

    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "python_output"
            / f"{args.dataset_label}_steps_{args.start_step:04d}_{args.end_step:04d}"
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

    print(f"=== {args.dataset_label} BEMD Python batch ===")
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

    selected_steps = list(
        step_numbers(args.start_step, args.end_step, args.step_stride, args.limit_steps)
    )
    if args.render_only:
        processed_results = list_result_paths(results_dir, selected_steps)
        print(f"Render-only mode: found {len(processed_results)} existing result files.")
    elif args.workers == 1:
        configure_worker_thread_env(args.threads_per_worker)
        bemd_func = configure_bemd(args)
        processed_results = run_steps_serial(args, steps_to_process, bemd_func, results_dir, error_log)
    else:
        processed_results = run_steps_parallel(args, steps_to_process, results_dir, error_log)

    if missing_steps:
        with (args.output_dir / "missing_steps.json").open("w", encoding="utf-8") as handle:
            json.dump(missing_steps, handle, indent=2)
        print(f"Skipped {len(missing_steps)} missing step files.")

    result_paths_for_viz = list_result_paths(results_dir, selected_steps)
    if not args.no_frames and result_paths_for_viz:
        color_limits = compute_color_limits(
            result_paths_for_viz,
            args.color_scale,
            args.color_percentile,
            args.reconstruction_mode,
        )
        limits_path = save_color_limits(args.output_dir, color_limits)
        print(f"Fixed color limits ({args.color_scale}) saved to {limits_path}")
        for key, value in color_limits.to_dict().items():
            if key.endswith("_vmax") or key.endswith("_sym"):
                print(f"  {key}: {value}")
        render_all_frames(args, result_paths_for_viz, color_limits)

    if not args.no_video and not args.no_frames:
        write_video(
            frame_dir(args.output_dir, "summary", args.reconstruction_mode),
            video_path(
                args.output_dir,
                args.dataset_label,
                args.start_step,
                args.end_step,
                "summary",
                args.reconstruction_mode,
            ),
            args.fps,
        )
        write_video(
            frame_dir(args.output_dir, "e_imfs", args.reconstruction_mode),
            video_path(
                args.output_dir,
                args.dataset_label,
                args.start_step,
                args.end_step,
                "E_imfs",
                args.reconstruction_mode,
            ),
            args.fps,
        )

    print(f"Processed or reused {len(processed_results)} result files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
