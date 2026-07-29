#!/usr/bin/env python3
"""自动运行全窗口特征提取与独立验证 SVM 实验。

一条命令运行完整流程：
    python run_svm_experiments.py

默认固定预处理参数：
    source rate:    500 Hz
    downsampling D: 1, 2
    BPF:            20--min(100, 0.45*effective Fs) Hz,
                    total-4th-order Butterworth
    reference win:  128 source samples
    effective FFT:  128/D points
    overlap:        50%（hop = 64/D）
    transition band: button 上升沿/下降沿前后各 50 ms，仅审计、不丢窗

因此默认运行2个D乘4种特征组合，共8组实验。D改变时，名义窗口时长仍为
256 ms，更新周期仍为128 ms，FFT频率分辨率仍为3.90625 Hz；滤波群延迟
和实际处理时间仍需在硬件上测量。

训练数据仅来自 ``data_collection/state_train``，独立验证数据仅来自
``data_collection/state_validation``。脚本不会随机拆分重叠窗口。
"""

from __future__ import annotations

import argparse
import csv
import json
import locale
import subprocess
import sys
import time
from pathlib import Path


# 四组实验分别测量纯时域、纯频域以及时频域组合的效果与部署成本。
DEFAULT_FEATURE_SETS = {
    "TD3": ["mav", "rms", "wl"],
    "TD5": ["mav", "rms", "wl", "zc", "ssc"],
    "FD4": ["mnf", "mdf", "pkf", "bandpower"],
    "TD_FD9": [
        "mav", "rms", "wl", "zc", "ssc",
        "mnf", "mdf", "pkf", "bandpower",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate sEMG feature extraction and Linear-SVM training."
    )
    parser.add_argument(
        "--train-input-dir",
        type=Path,
        default=Path("data_collection/state_train"),
    )
    parser.add_argument(
        "--validation-input-dir",
        type=Path,
        default=Path("data_collection/state_validation"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiment_results_state")
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=500,
        help="Source ADC sampling rate before digital downsampling.",
    )
    parser.add_argument(
        "--downsample-factors",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Integer D values; default runs D=1 and D=2.",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=128,
        help="Reference window length at the source sampling rate.",
    )
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--transition-band-ms",
        "--edge-guard-ms",
        dest="transition_band_ms",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--clench-fraction-threshold",
        type=float,
        default=0.5,
        help=(
            "Window label is 1 only when its fraction of source-sample "
            "Clench labels is strictly greater than this value."
        ),
    )
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--aa-passband-ripple-db", type=float, default=1.0)
    parser.add_argument(
        "--aa-stopband-attenuation-db", type=float, default=40.0
    )
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only TD3 with the first D for a quick pipeline check.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed experiment and continue with the remaining grid.",
    )
    args = parser.parse_args()
    if args.sampling_rate <= 0:
        parser.error("--sampling-rate must be positive.")
    if not args.downsample_factors:
        parser.error("--downsample-factors requires at least one D.")
    if any(factor < 1 for factor in args.downsample_factors):
        parser.error("Every --downsample-factors value must be >= 1.")
    if len(set(args.downsample_factors)) != len(args.downsample_factors):
        parser.error("--downsample-factors cannot contain duplicates.")
    if args.window_samples < 2:
        parser.error("--window-samples must be at least 2.")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must satisfy 0 <= overlap < 1.")
    if args.transition_band_ms < 0:
        parser.error("--transition-band-ms cannot be negative.")
    if not 0.0 <= args.clench_fraction_threshold < 1.0:
        parser.error(
            "--clench-fraction-threshold must satisfy 0 <= threshold < 1."
        )
    reference_hop = round(args.window_samples * (1.0 - args.overlap))
    if reference_hop < 1:
        parser.error("Window/overlap produces an invalid reference hop.")
    for factor in args.downsample_factors:
        if args.window_samples % factor or reference_hop % factor:
            parser.error(
                f"D={factor} must divide both reference window "
                f"({args.window_samples}) and reference hop "
                f"({reference_hop}) exactly."
            )
        effective_sampling_rate = args.sampling_rate / factor
        effective_high_hz = min(
            args.high_hz, 0.45 * effective_sampling_rate
        )
        if not 0 < args.low_hz < effective_high_hz:
            parser.error(
                f"For D={factor}, require 0 < low-hz < "
                "min(high-hz, 0.45 * effective sampling rate); "
                f"effective high is {effective_high_hz:g} Hz."
            )
    if args.filter_order < 2 or args.filter_order % 2:
        parser.error("--filter-order must be an even integer >= 2.")
    if args.c <= 0:
        parser.error("--c must be positive.")
    if args.aa_passband_ripple_db <= 0:
        parser.error("--aa-passband-ripple-db must be positive.")
    if (
        args.aa_stopband_attenuation_db
        <= args.aa_passband_ripple_db
    ):
        parser.error(
            "--aa-stopband-attenuation-db must exceed "
            "--aa-passband-ripple-db."
        )
    args.reference_hop_samples = reference_hop
    return args


def run_command(command: list[str], log_path: Path) -> None:
    """Run one stage and save both its command and terminal output."""
    result = subprocess.run(
        command,
        text=True,
        # Windows下子Python进程连接到PIPE时通常使用系统代码页（中文环境为
        # cp936/GBK），不能固定按UTF-8解码，否则日志和终端输出可能报错。
        encoding=locale.getpreferredencoding(False),
        errors="backslashreplace",
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
    feature_set: str,
    source_sampling_rate: int,
    downsample_factor: int,
    reference_window_samples: int,
    overlap: float,
    low_hz: float,
    requested_high_hz: float,
    effective_high_hz: float,
    filter_order: int,
    transition_band_ms: float,
    clench_fraction_threshold: float,
    aa_passband_ripple_db: float,
    aa_stopband_attenuation_db: float,
    c_value: float,
) -> str:
    overlap_percent = round(overlap * 100)
    effective_sampling_rate = source_sampling_rate / downsample_factor
    effective_window_samples = (
        reference_window_samples // downsample_factor
    )
    fs_text = f"{effective_sampling_rate:g}".replace(".", "p")
    parameter_text = (
        f"bpf{low_hz:g}-{requested_high_hz:g}to"
        f"{effective_high_hz:g}_o{filter_order}_"
        f"band{transition_band_ms:g}_"
        f"lt{clench_fraction_threshold:g}_"
        f"aa{aa_passband_ripple_db:g}-{aa_stopband_attenuation_db:g}_"
        f"c{c_value:g}"
    ).replace(".", "p")
    return (
        f"{feature_set.lower()}_d{downsample_factor}_fs{fs_text}_"
        f"fft{effective_window_samples}_ov{overlap_percent}_"
        f"{parameter_text}"
    )


def write_summary(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] != "ok"]
    successful.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    ordered = successful + failed
    fieldnames = [
        "rank", "status", "experiment", "feature_set",
        "source_sampling_rate", "downsample_factor",
        "effective_sampling_rate",
        "reference_window_samples", "window_samples", "fft_points",
        "source_hop_samples", "hop_samples",
        "window_duration_ms", "update_period_ms",
        "overlap", "transition_band_ms", "clench_fraction_threshold",
        "low_hz", "high_hz", "effective_high_hz", "filter_order",
        "aa_passband_ripple_db", "aa_stopband_attenuation_db",
        "features", "feature_count",
        "train_windows", "train_sessions",
        "validation_windows", "validation_sessions",
        "accuracy", "balanced_accuracy", "f1",
        "outside_band_windows", "outside_band_accuracy",
        "outside_band_balanced_accuracy",
        "transition_band_windows", "transition_band_accuracy",
        "transition_band_balanced_accuracy",
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

    # 训练与验证必须来自两个独立目录，不能把同一 session 的重叠窗口随机拆开。
    train_dir = args.train_input_dir.resolve()
    validation_dir = args.validation_input_dir.resolve()
    if train_dir == validation_dir:
        raise ValueError(
            "Training and validation input directories must be different."
        )

    if args.quick:
        configurations = [
            (
                args.downsample_factors[0],
                "TD3",
                DEFAULT_FEATURE_SETS["TD3"],
            )
        ]
    else:
        configurations = [
            (factor, feature_set, features)
            for factor in args.downsample_factors
            for feature_set, features in DEFAULT_FEATURE_SETS.items()
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "experiment_summary.csv"
    feature_cache_dir = args.output_dir / "_feature_cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_downsample_factors: set[int] = set()
    rows: list[dict] = []
    total = len(configurations)
    print(f"Starting {total} experiment(s). Output: {args.output_dir.resolve()}")

    for number, (downsample_factor, feature_set, features) in enumerate(
        configurations, start=1
    ):
        effective_sampling_rate = (
            args.sampling_rate / downsample_factor
        )
        effective_high_hz = min(
            args.high_hz, 0.45 * effective_sampling_rate
        )
        effective_window_samples = (
            args.window_samples // downsample_factor
        )
        effective_hop_samples = (
            args.reference_hop_samples // downsample_factor
        )
        window_duration_ms = (
            1000.0 * effective_window_samples / effective_sampling_rate
        )
        update_period_ms = (
            1000.0 * effective_hop_samples / effective_sampling_rate
        )
        name = experiment_name(
            feature_set,
            args.sampling_rate,
            downsample_factor,
            args.window_samples,
            args.overlap,
            args.low_hz,
            args.high_hz,
            effective_high_hz,
            args.filter_order,
            args.transition_band_ms,
            args.clench_fraction_threshold,
            args.aa_passband_ripple_db,
            args.aa_stopband_attenuation_db,
            args.c,
        )
        experiment_dir = args.output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        # 四组实验采用完全相同的预处理和窗口，因此只提取一次全部九个特征。
        # 各模型训练时再按照 TD3、TD5、FD4 或 TD_FD9 选择对应列。
        overlap_percent = round(args.overlap * 100)
        effective_fs_text = f"{effective_sampling_rate:g}".replace(".", "p")
        preprocessing_name = (
            f"srcfs{args.sampling_rate}_d{downsample_factor}_"
            f"fs{effective_fs_text}_"
            f"win{args.window_samples}to{effective_window_samples}_"
            f"hop{args.reference_hop_samples}to{effective_hop_samples}_"
            f"ov{overlap_percent}_band{args.transition_band_ms:g}_"
            f"labelthr{args.clench_fraction_threshold:g}_"
            f"bpf{args.low_hz:g}-{args.high_hz:g}to"
            f"{effective_high_hz:g}_o{args.filter_order}_"
            f"aa{args.aa_passband_ripple_db:g}-"
            f"{args.aa_stopband_attenuation_db:g}"
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
        all_features = DEFAULT_FEATURE_SETS["TD_FD9"]
        train_extract_command = [
            sys.executable, str(extractor),
            "--input-dir", str(args.train_input_dir),
            "--output", str(train_feature_path),
            "--sampling-rate", str(args.sampling_rate),
            "--downsample-factor", str(downsample_factor),
            "--window-samples", str(args.window_samples),
            "--overlap", str(args.overlap),
            "--transition-band-ms", str(args.transition_band_ms),
            "--clench-fraction-threshold",
            str(args.clench_fraction_threshold),
            "--low-hz", str(args.low_hz),
            "--high-hz", str(args.high_hz),
            "--fft-low-hz", str(args.low_hz),
            "--fft-high-hz", str(args.high_hz),
            "--filter-order", str(args.filter_order),
            "--aa-passband-ripple-db", str(args.aa_passband_ripple_db),
            "--aa-stopband-attenuation-db",
            str(args.aa_stopband_attenuation_db),
            "--features", *all_features,
        ]
        validation_extract_command = [
            sys.executable, str(extractor),
            "--input-dir", str(args.validation_input_dir),
            "--output", str(validation_feature_path),
            "--sampling-rate", str(args.sampling_rate),
            "--downsample-factor", str(downsample_factor),
            "--window-samples", str(args.window_samples),
            "--overlap", str(args.overlap),
            "--transition-band-ms", str(args.transition_band_ms),
            "--clench-fraction-threshold",
            str(args.clench_fraction_threshold),
            "--low-hz", str(args.low_hz),
            "--high-hz", str(args.high_hz),
            "--fft-low-hz", str(args.low_hz),
            "--fft-high-hz", str(args.high_hz),
            "--filter-order", str(args.filter_order),
            "--aa-passband-ripple-db", str(args.aa_passband_ripple_db),
            "--aa-stopband-attenuation-db",
            str(args.aa_stopband_attenuation_db),
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
        common_result_fields = {
            "experiment": name,
            "feature_set": feature_set,
            "source_sampling_rate": args.sampling_rate,
            "downsample_factor": downsample_factor,
            "effective_sampling_rate": effective_sampling_rate,
            "reference_window_samples": args.window_samples,
            "window_samples": effective_window_samples,
            "fft_points": effective_window_samples,
            "source_hop_samples": args.reference_hop_samples,
            "hop_samples": effective_hop_samples,
            "window_duration_ms": window_duration_ms,
            "update_period_ms": update_period_ms,
            "overlap": args.overlap,
            "transition_band_ms": args.transition_band_ms,
            "clench_fraction_threshold": args.clench_fraction_threshold,
            "low_hz": args.low_hz,
            "high_hz": args.high_hz,
            "effective_high_hz": effective_high_hz,
            "filter_order": args.filter_order,
            "aa_passband_ripple_db": args.aa_passband_ripple_db,
            "aa_stopband_attenuation_db": (
                args.aa_stopband_attenuation_db
            ),
            "features": "+".join(features),
            "feature_count": len(features),
        }

        try:
            if downsample_factor not in extracted_downsample_factors:
                run_command(
                    train_extract_command,
                    cached_dir / "train_extract.log",
                )
                run_command(
                    validation_extract_command,
                    cached_dir / "validation_extract.log",
                )
                extracted_downsample_factors.add(downsample_factor)
            run_command(train_command, experiment_dir / "train.log")
            run_command(
                validation_command,
                experiment_dir / "validation.log",
            )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            matrix = metrics["confusion_matrix"]
            band_breakdown = metrics["transition_band_breakdown"]
            outside_band = band_breakdown["outside_transition_band"]
            inside_band = band_breakdown["intersects_transition_band"]
            rows.append({
                "status": "ok",
                **common_result_fields,
                "train_windows": metrics["train_windows"],
                "train_sessions": metrics["train_sessions"],
                "validation_windows": metrics["validation_windows"],
                "validation_sessions": metrics["validation_sessions"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "outside_band_windows": (
                    outside_band["windows"] if outside_band else 0
                ),
                "outside_band_accuracy": (
                    outside_band["accuracy"] if outside_band else None
                ),
                "outside_band_balanced_accuracy": (
                    outside_band["balanced_accuracy"]
                    if outside_band
                    else None
                ),
                "transition_band_windows": (
                    inside_band["windows"] if inside_band else 0
                ),
                "transition_band_accuracy": (
                    inside_band["accuracy"] if inside_band else None
                ),
                "transition_band_balanced_accuracy": (
                    inside_band["balanced_accuracy"]
                    if inside_band
                    else None
                ),
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
                **common_result_fields,
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
