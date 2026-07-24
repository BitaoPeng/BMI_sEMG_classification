#!/usr/bin/env python3
"""Run feature extraction and grouped Linear-SVM experiments automatically.

One command runs the complete pipeline:
    python run_svm_experiments.py

Default grid:
    target sampling rates: 500, 250, 125 Hz
    window lengths:        100, 200, 300 ms
    overlap:               50%
    feature sets:          4 combinations

This creates 36 experiments under ``experiment_results`` and an aggregate
``experiment_summary.csv`` sorted by mean balanced accuracy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


# Edit these lists if a different default experiment grid is wanted.
DEFAULT_TARGET_FS = [500, 250, 125]
DEFAULT_WINDOW_MS = [100, 200, 300]
DEFAULT_OVERLAPS = [0.5]
DEFAULT_FEATURE_SETS = [
    ["mav"],
    ["mav", "wl"],
    ["mav", "rms", "wl"],
    ["mav", "wl", "zc", "ssc"],
]


def comma_ints(value: str) -> list[int]:
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc


def comma_floats(value: str) -> list[float]:
    try:
        return [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate sEMG feature extraction and Linear-SVM training."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data_saved/data_saved")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiment_results")
    )
    parser.add_argument("--source-fs", type=int, default=500)
    parser.add_argument(
        "--target-fs", type=comma_ints, default=DEFAULT_TARGET_FS,
        help="Comma-separated target rates, e.g. 500,250,125."
    )
    parser.add_argument(
        "--window-ms", type=comma_ints, default=DEFAULT_WINDOW_MS,
        help="Comma-separated window lengths, e.g. 100,200,300."
    )
    parser.add_argument(
        "--overlaps", type=comma_floats, default=DEFAULT_OVERLAPS,
        help="Comma-separated overlap fractions, e.g. 0,0.5,0.75."
    )
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)
    parser.add_argument("--notch-hz", type=float, default=50.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--discard-ms", type=float, default=300.0)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only one baseline configuration for a quick pipeline check.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed experiment and continue with the remaining grid.",
    )
    return parser.parse_args()


def run_command(command: list[str], log_path: Path) -> None:
    """Run one stage and save both its command and terminal output."""
    result = subprocess.run(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    command_text = subprocess.list2cmdline(command)
    log_path.write_text(
        f"COMMAND\n{command_text}\n\nOUTPUT\n{result.stdout}",
        encoding="utf-8",
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; see {log_path}"
        )


def experiment_name(
    target_fs: int,
    window_ms: int,
    overlap: float,
    features: list[str],
) -> str:
    overlap_percent = round(overlap * 100)
    feature_text = "-".join(features)
    return (
        f"fs{target_fs}_win{window_ms}_ov{overlap_percent}_"
        f"{feature_text}"
    )


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    successful.sort(
        key=lambda row: row["balanced_accuracy_mean"], reverse=True
    )
    ordered = successful + failed
    fieldnames = [
        "rank", "status", "experiment", "target_fs", "window_ms",
        "overlap", "features", "feature_count", "windows", "trials",
        "accuracy_mean", "accuracy_std", "balanced_accuracy_mean",
        "balanced_accuracy_std", "f1_mean", "f1_std",
        "tn", "fp", "fn", "tp", "elapsed_seconds", "error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(ordered, start=1):
            output = {key: row.get(key, "") for key in fieldnames}
            output["rank"] = index if row["status"] == "ok" else ""
            writer.writerow(output)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    extractor = script_dir / "emg_feature_extractor.py"
    trainer = script_dir / "train_linear_svm.py"
    if not extractor.exists() or not trainer.exists():
        raise FileNotFoundError(
            "emg_feature_extractor.py and train_linear_svm.py must be beside "
            "this script."
        )

    if args.quick:
        configurations = [(250, 200, 0.5, ["mav", "wl", "zc", "ssc"])]
    else:
        configurations = list(itertools.product(
            args.target_fs,
            args.window_ms,
            args.overlaps,
            DEFAULT_FEATURE_SETS,
        ))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "experiment_summary.csv"
    rows: list[dict] = []
    total = len(configurations)
    print(f"Starting {total} experiment(s). Output: {args.output_dir.resolve()}")

    for number, (target_fs, window_ms, overlap, features) in enumerate(
        configurations, start=1
    ):
        name = experiment_name(target_fs, window_ms, overlap, features)
        experiment_dir = args.output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        feature_path = experiment_dir / "features.csv"
        model_path = experiment_dir / "linear_svm.joblib"
        params_path = experiment_dir / "linear_svm_params.json"
        metrics_path = experiment_dir / "metrics.json"
        start_time = time.monotonic()
        print(f"\n[{number}/{total}] {name}")

        extract_command = [
            sys.executable, str(extractor),
            "--input-dir", str(args.input_dir),
            "--output", str(feature_path),
            "--source-fs", str(args.source_fs),
            "--target-fs", str(target_fs),
            "--window-ms", str(window_ms),
            "--discard-ms", str(args.discard_ms),
            "--overlap", str(overlap),
            "--low-hz", str(args.low_hz),
            "--high-hz", str(args.high_hz),
            "--notch-hz", str(args.notch_hz),
            "--filter-order", str(args.filter_order),
            "--threshold", str(args.threshold),
            "--features", *features,
        ]
        train_command = [
            sys.executable, str(trainer),
            "--input", str(feature_path),
            "--features", *features,
            "--c", str(args.c),
            "--folds", str(args.folds),
            "--seed", str(args.seed),
            "--model-output", str(model_path),
            "--params-output", str(params_path),
            "--metrics-output", str(metrics_path),
        ]

        try:
            run_command(extract_command, experiment_dir / "extract.log")
            run_command(train_command, experiment_dir / "train.log")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            matrix = metrics["confusion_matrix"]
            rows.append({
                "status": "ok",
                "experiment": name,
                "target_fs": target_fs,
                "window_ms": window_ms,
                "overlap": overlap,
                "features": "+".join(features),
                "feature_count": len(features),
                "windows": metrics["windows"],
                "trials": metrics["trials"],
                "accuracy_mean": metrics["accuracy_mean"],
                "accuracy_std": metrics["accuracy_std"],
                "balanced_accuracy_mean": metrics[
                    "balanced_accuracy_mean"
                ],
                "balanced_accuracy_std": metrics[
                    "balanced_accuracy_std"
                ],
                "f1_mean": metrics["f1_mean"],
                "f1_std": metrics["f1_std"],
                "tn": matrix[0][0],
                "fp": matrix[0][1],
                "fn": matrix[1][0],
                "tp": matrix[1][1],
                "elapsed_seconds": round(time.monotonic() - start_time, 3),
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "status": "failed",
                "experiment": name,
                "target_fs": target_fs,
                "window_ms": window_ms,
                "overlap": overlap,
                "features": "+".join(features),
                "feature_count": len(features),
                "elapsed_seconds": round(time.monotonic() - start_time, 3),
                "error": str(exc),
            })
            write_summary(rows, summary_path)
            if not args.continue_on_error:
                raise
            print(f"FAILED: {exc}")

        # Keep a usable partial summary even if a later experiment is stopped.
        write_summary(rows, summary_path)

    successful = sum(row["status"] == "ok" for row in rows)
    print(f"\nCompleted: {successful}/{total} successful experiments")
    print(f"Summary: {summary_path.resolve()}")
    if successful:
        best = max(
            (row for row in rows if row["status"] == "ok"),
            key=lambda row: row["balanced_accuracy_mean"],
        )
        print(
            "Best mean balanced accuracy: "
            f"{best['balanced_accuracy_mean']:.4f} ({best['experiment']})"
        )
    return 0 if successful == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
