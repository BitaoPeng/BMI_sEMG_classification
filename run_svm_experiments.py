#!/usr/bin/env python3
"""Run feature extraction and independent-validation SVM experiments.

One command runs the complete pipeline:
    python run_svm_experiments.py

Default grid:
    target sampling rates: 500, 250, 125 Hz
    window lengths:        100, 200, 300 ms
    overlap:               50%
    feature sets:          4 combinations

``data_collection/data_saved_01`` is used only for training and
``data_collection/data_saved_02`` only for independent validation. This
creates 36 experiments under
``experiment_results`` and an aggregate ``experiment_summary.csv`` sorted by
validation balanced accuracy.
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
        "--train-input-dir",
        type=Path,
        default=Path(
            "data_collection/data_saved_01/data_saved_01"
        ),
    )
    parser.add_argument(
        "--validation-input-dir",
        type=Path,
        default=Path(
            "data_collection/data_saved_02/data_saved_02"
        ),
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
    # 每个 CSV 的前 0.5 s 不生成窗口，也不会进入训练或验证标签。
    parser.add_argument("--discard-ms", type=float, default=500.0)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--c", type=float, default=1.0)
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
    successful.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    ordered = successful + failed
    fieldnames = [
        "rank", "status", "experiment", "target_fs", "window_ms",
        "overlap", "features", "feature_count",
        "train_windows", "train_trials",
        "validation_windows", "validation_trials",
        "accuracy", "balanced_accuracy", "f1",
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
    trainer = script_dir / "train_svm.py"
    validator = script_dir / "validate_svm.py"
    if not extractor.exists() or not trainer.exists() or not validator.exists():
        raise FileNotFoundError(
            "emg_feature_extractor.py, train_svm.py and validate_svm.py "
            "must be beside this script."
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
    feature_cache_dir = args.output_dir / "_feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_configurations: set[tuple[int, int, float]] = set()
    rows: list[dict] = []
    total = len(configurations)
    print(f"Starting {total} experiment(s). Output: {args.output_dir.resolve()}")

    for number, (target_fs, window_ms, overlap, features) in enumerate(
        configurations, start=1
    ):
        name = experiment_name(target_fs, window_ms, overlap, features)
        experiment_dir = args.output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # 同一个 sampling rate/window/overlap 的原始预处理结果可以供不同
        # feature combinations 共用，避免重复进行四次滤波和窗口划分。
        preprocessing_key = (target_fs, window_ms, overlap)
        preprocessing_name = (
            f"fs{target_fs}_win{window_ms}_ov{round(overlap * 100)}"
        )
        cached_dir = feature_cache_dir / preprocessing_name
        cached_dir.mkdir(parents=True, exist_ok=True)
        train_feature_path = cached_dir / "train_features.csv"
        validation_feature_path = cached_dir / "validation_features.csv"
        model_path = experiment_dir / "linear_svm.joblib"
        params_path = experiment_dir / "linear_svm_params.json"
        metrics_path = experiment_dir / "metrics.json"
        start_time = time.monotonic()
        print(f"\n[{number}/{total}] {name}")

        # 一次提取所有候选特征，训练时再选择本实验需要的特征列。
        all_features = sorted({
            feature
            for feature_set in DEFAULT_FEATURE_SETS
            for feature in feature_set
        })
        train_extract_command = [
            sys.executable, str(extractor),
            "--input-dir", str(args.train_input_dir),
            "--output", str(train_feature_path),
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
            "--features", *all_features,
        ]
        validation_extract_command = [
            sys.executable, str(extractor),
            "--input-dir", str(args.validation_input_dir),
            "--output", str(validation_feature_path),
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
            "--features", *all_features,
        ]
        train_command = [
            sys.executable, str(trainer),
            "--input", str(train_feature_path),
            "--features", *features,
            "--c", str(args.c),
            "--model-output", str(model_path),
            "--params-output", str(params_path),
        ]
        validation_command = [
            sys.executable, str(validator),
            "--model-input", str(model_path),
            "--validation-input", str(validation_feature_path),
            "--metrics-output", str(metrics_path),
        ]

        try:
            if preprocessing_key not in extracted_configurations:
                run_command(
                    train_extract_command,
                    cached_dir / "train_extract.log",
                )
                run_command(
                    validation_extract_command,
                    cached_dir / "validation_extract.log",
                )
                extracted_configurations.add(preprocessing_key)
            run_command(train_command, experiment_dir / "train.log")
            run_command(
                validation_command,
                experiment_dir / "validation.log",
            )
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
                "train_windows": metrics["train_windows"],
                "train_trials": metrics["train_trials"],
                "validation_windows": metrics["validation_windows"],
                "validation_trials": metrics["validation_trials"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
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
            key=lambda row: row["balanced_accuracy"],
        )
        print(
            "Best validation balanced accuracy: "
            f"{best['balanced_accuracy']:.4f} ({best['experiment']})"
        )
    return 0 if successful == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
