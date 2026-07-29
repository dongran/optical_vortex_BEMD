#!/usr/bin/env python3
"""Re-decompose saved raw fields with strict MATLAB-compatible BEMD settings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "bemd_nn_project/data/manifest/manifest.jsonl"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "python_output"
BEMD_ROOT = REPO_ROOT / "BEMD_Python"
COMPONENTS = ("V1", "V2", "V3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loam", type=int, action="append", required=True)
    parser.add_argument("--start-step", type=int, default=600)
    parser.add_argument("--end-step", type=int, default=1999)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument("--crop-size", type=int, default=0)
    parser.add_argument("--nimfs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--sift-cost-threshold", type=float, default=0.2)
    parser.add_argument("--gridfit-smoothness", type=float, default=1.0)
    parser.add_argument(
        "--linear-solver",
        choices=("backslash", "normal"),
        default="backslash",
        help=(
            "backslash is the strict LSQR reference; normal is the fast equivalent "
            "for batches after checking its error against the reference."
        ),
    )
    parser.add_argument("--decompose-e", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record strict extrema/numerical failures and continue the remaining frames.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = set(args.loam)
    rows: list[dict[str, Any]] = []
    with resolve_repo_path(args.manifest).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            loam = int(row["loam"])
            step = int(row["step"])
            if loam not in requested:
                continue
            if step < args.start_step or step > args.end_step:
                continue
            if (step - args.start_step) % args.step_stride != 0:
                continue
            source = Path(row["path"])
            if not source.exists():
                source = REPO_ROOT / source.relative_to("/home/randong/Learning_HHT_FDTD_simulation")
            if not source.exists():
                raise FileNotFoundError(source)
            rows.append({"loam": loam, "step": step, "source": str(source)})

    rows.sort(key=lambda row: (row["loam"], row["step"]))
    if args.limit_steps is not None:
        limited: list[dict[str, Any]] = []
        for loam in sorted(requested):
            limited.extend(
                [row for row in rows if row["loam"] == loam][: args.limit_steps]
            )
        rows = limited
    if not rows:
        raise ValueError("No source NPZ files matched the requested range.")
    return rows


def center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0 or crop_size >= min(array.shape[:2]):
        return np.asarray(array)
    top = (array.shape[0] - crop_size) // 2
    left = (array.shape[1] - crop_size) // 2
    return np.asarray(array[top : top + crop_size, left : left + crop_size])


def worker_init(threads: int) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(max(1, threads))
    if str(BEMD_ROOT) not in sys.path:
        sys.path.insert(0, str(BEMD_ROOT))


def process_one(task: dict[str, Any]) -> dict[str, Any]:
    from bemd import bemd

    source = Path(task["source"])
    output = Path(task["output"])
    if output.exists() and not task["overwrite"]:
        return {"status": "skipped", "output": str(output), **task}

    started = time.time()
    with np.load(source, allow_pickle=False) as loaded:
        fields = {
            name: center_crop(np.asarray(loaded[name], dtype=np.float64), task["crop_size"])
            for name in ("E", *COMPONENTS)
        }

    imfs: dict[str, np.ndarray] = {}
    names = ("E", *COMPONENTS) if task["decompose_e"] else COMPONENTS
    for name in names:
        kwargs = {
            "cost_threshold": task["cost_threshold"],
            "gridfit_smoothness": task["smoothness"],
        }
        if task["linear_solver"] == "backslash":
            kwargs["matlab_compatible"] = True
        else:
            kwargs.update(
                max_iterations=None,
                gridfit_solver="normal",
                insufficient_extrema="raise",
                nan_policy="raise",
            )
        imfs[name] = bemd(fields[name], task["nimfs"], **kwargs).astype(
            np.float32, copy=False
        )

    metadata = {
        "step": task["step"],
        "loam": task["loam"],
        "source_npz": str(source),
        "cropped_shape": list(fields["E"].shape),
        "nimfs": task["nimfs"],
        "sift_cost_threshold": task["cost_threshold"],
        "sift_max_iterations": None,
        "gridfit_smoothness": task["smoothness"],
        "gridfit_solver": task["linear_solver"],
        "insufficient_extrema": "raise",
        "nan_policy": "raise",
        "matlab_compatible": task["linear_solver"] == "backslash",
        "matlab_compatible_policies": True,
        "elapsed_seconds": time.time() - started,
    }
    payload: dict[str, Any] = {
        name: value.astype(np.float32, copy=False) for name, value in fields.items()
    }
    payload.update({f"imf_{name}": value for name, value in imfs.items()})
    payload["metadata_json"] = np.asarray(json.dumps(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    save = np.savez_compressed if task["compressed"] else np.savez
    save(temporary, **payload)
    temporary.replace(output)
    return {"status": "written", "output": str(output), **task, **metadata}


def main() -> int:
    args = parse_args()
    if args.nimfs < 2:
        raise ValueError("--nimfs must be at least 2.")
    if args.step_stride < 1:
        raise ValueError("--step-stride must be positive.")

    output_dir = resolve_repo_path(args.output_dir)
    rows = load_rows(args)
    tasks = [
        {
            **row,
            "output": str(
                output_dir / f"loam{row['loam']}" / "results" / f"step_{row['step']:04d}.npz"
            ),
            "crop_size": args.crop_size,
            "nimfs": args.nimfs,
            "cost_threshold": args.sift_cost_threshold,
            "smoothness": args.gridfit_smoothness,
            "linear_solver": args.linear_solver,
            "decompose_e": bool(args.decompose_e),
            "compressed": bool(args.compressed),
            "overwrite": bool(args.overwrite),
        }
        for row in rows
    ]

    print(
        f"Strict MATLAB-compatible BEMD: {len(tasks)} frames, "
        f"loam={sorted(set(args.loam))}, workers={args.workers}"
    )
    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        worker_init(args.threads_per_worker)
        for index, task in enumerate(tasks, start=1):
            try:
                record = process_one(task)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                record = {
                    **task,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(record)
            print(
                f"[{index}/{len(tasks)}] {record['status']} {record['output']} "
                f"{record.get('error', '')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(args.threads_per_worker,),
        ) as pool:
            futures = {pool.submit(process_one, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    record = {
                        **task,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
                print(
                    f"[{index}/{len(tasks)}] {record['status']} {record['output']} "
                    f"{record.get('error', '')}",
                    flush=True,
                )

    records.sort(key=lambda row: (row["loam"], row["step"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
