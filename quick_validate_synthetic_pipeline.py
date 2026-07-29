#!/usr/bin/env python3
"""用带按钮标签错位的合成 sEMG 快速回归验证完整流水线。

这个脚本解决“还没有新的按钮标注实测数据，但想先确认代码接口和逻辑正确”
的问题。它会：

1. 生成连续的 500 Hz ``Channel1_raw``，幅值范围参考项目中的旧实测数据；
2. 生成隐藏真值 ``sim_true_label``（真实 Relax/Clench 状态）；
3. 让 GUI ``button_label`` 的每个按下/松开边沿相对真值随机提前或滞后；
4. 默认混合 100/200/500/2000 ms 的握拳动作，并随机改变动作顺序和间隔；
5. 调用现有 ``run_svm_experiments.py``，分别快速测试 D=1 和 D=2；
6. 自动检查 transition band、全窗口标注、NFFT、BPF、FFT、SVM 参数导出。

合成数据只用于验证代码（software regression test），不能代替真实受试者数据，
也不能把合成数据准确率写成最终实验性能。

运行：

    python quick_validate_synthetic_pipeline.py

只生成模拟 CSV，不运行训练：

    python quick_validate_synthetic_pipeline.py --generate-only
"""

from __future__ import annotations

import argparse
import json
import locale
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, sosfreqz


SOURCE_FS = 500
REFERENCE_WINDOW_SAMPLES = 128
REFERENCE_HOP_SAMPLES = 64
REQUESTED_BPF_LOW_HZ = 20.0
REQUESTED_BPF_HIGH_HZ = 100.0
ALL_FEATURES = [
    "mav",
    "rms",
    "wl",
    "zc",
    "ssc",
    "mnf",
    "mdf",
    "pkf",
    "bandpower",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate button-misaligned synthetic sEMG sessions and quickly "
            "validate the state-classification pipeline."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("quick_validation_artifacts"),
        help="Each run creates a new timestamped folder below this directory.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--train-sessions", type=int, default=3)
    parser.add_argument("--validation-sessions", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=18.0)
    parser.add_argument(
        "--clench-durations-ms",
        type=float,
        nargs="+",
        default=[100.0, 200.0, 500.0, 2000.0],
        metavar="MS",
        help=(
            "Clench durations mixed in every session. Default: "
            "100 200 500 2000 ms."
        ),
    )
    parser.add_argument(
        "--button-jitter-std-ms",
        type=float,
        default=45.0,
        help="Standard deviation of button-edge timing error.",
    )
    parser.add_argument(
        "--button-jitter-max-ms",
        type=float,
        default=100.0,
        help="Absolute clipping limit of button-edge timing error.",
    )
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
    )
    parser.add_argument(
        "--minimum-balanced-accuracy",
        type=float,
        default=0.75,
        help="Regression-test floor for the deliberately separable simulation.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate CSV/manifest only; skip extraction, SVM and assertions.",
    )
    args = parser.parse_args()

    if args.train_sessions < 1 or args.validation_sessions < 1:
        parser.error("Session counts must both be >= 1.")
    if args.duration_s < 8.0:
        parser.error("--duration-s must be at least 8 seconds.")
    if not args.clench_durations_ms or any(
        duration_ms <= 0.0 for duration_ms in args.clench_durations_ms
    ):
        parser.error("--clench-durations-ms values must all be positive.")
    if sum(args.clench_durations_ms) / 1000.0 >= args.duration_s - 1.0:
        parser.error(
            "--duration-s is too short to contain every requested clench "
            "duration plus Relax intervals."
        )
    if args.button_jitter_std_ms < 0.0:
        parser.error("--button-jitter-std-ms cannot be negative.")
    if args.button_jitter_max_ms < args.button_jitter_std_ms:
        parser.error(
            "--button-jitter-max-ms must be >= --button-jitter-std-ms."
        )
    if args.transition_band_ms < 0.0:
        parser.error("--transition-band-ms cannot be negative.")
    if not 0.0 <= args.clench_fraction_threshold < 1.0:
        parser.error(
            "--clench-fraction-threshold must satisfy 0 <= threshold < 1."
        )
    if not 0.0 <= args.minimum_balanced_accuracy <= 1.0:
        parser.error("--minimum-balanced-accuracy must be between 0 and 1.")
    return args


def make_unique_run_dir(output_root: Path, seed: int) -> Path:
    """创建不覆盖旧结果的时间戳目录。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"run_{timestamp}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_true_state(
    sample_count: int,
    fs: int,
    clench_durations_ms: list[float],
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    list[dict[str, int]],
    list[dict[str, float | int | bool]],
]:
    """生成含多种握拳时长的真实状态、边沿和动作清单。"""
    state = np.zeros(sample_count, dtype=np.uint8)
    events: list[dict[str, int]] = []
    clench_bursts: list[dict[str, float | int | bool]] = []

    # 先保留一段随机Relax，避免文件一开始就是动作边沿。
    cursor = min(
        sample_count,
        max(1, int(round(float(rng.uniform(0.55, 0.90)) * fs))),
    )
    duration_cycle = np.asarray(clench_durations_ms, dtype=np.float64)

    while cursor < sample_count:
        # 每一轮都打乱顺序，保证有固定时长集合，同时避免机械重复。
        for requested_duration_ms in rng.permutation(duration_cycle):
            if cursor >= sample_count:
                break

            start = cursor
            requested_samples = max(
                1, int(round(float(requested_duration_ms) * fs / 1000.0))
            )
            stop = min(sample_count, start + requested_samples)
            state[start:stop] = 1
            events.append({"true_sample": int(start), "new_label": 1})
            complete = stop - start == requested_samples
            clench_bursts.append(
                {
                    "start_sample": int(start),
                    "stop_sample": int(stop),
                    "requested_duration_ms": float(requested_duration_ms),
                    "actual_duration_ms": 1000.0 * (stop - start) / fs,
                    "complete": bool(complete),
                }
            )
            cursor = stop

            if cursor >= sample_count:
                break
            events.append({"true_sample": int(cursor), "new_label": 0})

            # 每次握拳之间加入不同长度的Relax，模拟用户不规则操作。
            relax_samples = max(
                1, int(round(float(rng.uniform(0.55, 0.90)) * fs))
            )
            cursor = min(sample_count, cursor + relax_samples)

    complete_durations = {
        float(burst["requested_duration_ms"])
        for burst in clench_bursts
        if bool(burst["complete"])
    }
    missing_durations = set(map(float, clench_durations_ms)) - complete_durations
    if missing_durations:
        raise RuntimeError(
            "Session is too short to include every requested clench duration: "
            f"missing {sorted(missing_durations)} ms."
        )

    return state, events, clench_bursts


def make_button_label(
    true_events: list[dict[str, int]],
    sample_count: int,
    fs: int,
    jitter_std_ms: float,
    jitter_max_ms: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """给每个真实边沿加入随机提前/滞后，得到PC GUI按钮标签。"""
    button_label = np.zeros(sample_count, dtype=np.uint8)
    button_events: list[dict[str, float | int]] = []
    previous_button_sample = 0

    for event in true_events:
        requested_offset_ms = float(
            np.clip(
                rng.normal(0.0, jitter_std_ms),
                -jitter_max_ms,
                jitter_max_ms,
            )
        )
        offset_samples = int(round(requested_offset_ms * fs / 1000.0))
        button_sample = int(
            np.clip(
                int(event["true_sample"]) + offset_samples,
                previous_button_sample + 1,
                sample_count - 1,
            )
        )
        new_label = int(event["new_label"])
        button_label[button_sample:] = new_label
        actual_offset_ms = (
            1000.0
            * (button_sample - int(event["true_sample"]))
            / fs
        )
        button_events.append(
            {
                "true_sample": int(event["true_sample"]),
                "button_sample": button_sample,
                "new_label": new_label,
                "button_minus_true_ms": actual_offset_ms,
            }
        )
        previous_button_sample = button_sample

    return button_label, button_events


def make_activation_envelope(
    true_state: np.ndarray,
    fs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """用不同上升/下降时间常数模拟肌肉激活不是瞬间跳变。"""
    rise_tau_s = float(rng.uniform(0.050, 0.095))
    fall_tau_s = float(rng.uniform(0.070, 0.130))
    activation = np.zeros(true_state.size, dtype=np.float64)

    for index in range(1, true_state.size):
        target = float(true_state[index])
        tau_s = rise_tau_s if target > activation[index - 1] else fall_tau_s
        alpha = 1.0 - np.exp(-1.0 / (fs * tau_s))
        activation[index] = (
            activation[index - 1]
            + alpha * (target - activation[index - 1])
        )

    return activation, rise_tau_s, fall_tau_s


def normalized_band_noise(
    sample_count: int,
    fs: int,
    low_hz: float,
    high_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成单位标准差的带限随机载波，仅用于模拟原始sEMG。"""
    sos = butter(
        2,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    noise = sosfiltfilt(sos, rng.normal(size=sample_count))
    standard_deviation = float(np.std(noise))
    if standard_deviation <= np.finfo(np.float64).tiny:
        raise RuntimeError("Synthetic band-limited noise has zero variance.")
    return noise / standard_deviation


def make_motion_artifact(
    sample_count: int,
    fs: int,
    true_events: list[dict[str, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """在真实动作边沿附近加入短时低频运动伪迹和基线扰动。"""
    artifact = np.zeros(sample_count, dtype=np.float64)
    length = int(round(0.25 * fs))

    for event in true_events:
        start = int(event["true_sample"])
        stop = min(sample_count, start + length)
        count = stop - start
        if count <= 0:
            continue
        local_time = np.arange(count, dtype=np.float64) / fs
        amplitude = float(rng.uniform(5.0, 14.0))
        frequency_hz = float(rng.uniform(3.0, 8.0))
        sign = float(rng.choice([-1.0, 1.0]))
        artifact[start:stop] += (
            sign
            * amplitude
            * np.exp(-local_time / 0.080)
            * np.sin(2.0 * np.pi * frequency_hz * local_time)
        )

    return artifact


def generate_one_session(
    path: Path,
    split: str,
    session_index: int,
    duration_s: float,
    clench_durations_ms: list[float],
    fs: int,
    jitter_std_ms: float,
    jitter_max_ms: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """生成一个与当前feature extractor兼容的连续Session CSV。"""
    sample_count = int(round(duration_s * fs))
    time_s = np.arange(sample_count, dtype=np.float64) / fs
    true_state, true_events, clench_bursts = make_true_state(
        sample_count,
        fs,
        clench_durations_ms,
        rng,
    )
    button_label, button_events = make_button_label(
        true_events,
        sample_count,
        fs,
        jitter_std_ms,
        jitter_max_ms,
        rng,
    )
    activation, rise_tau_s, fall_tau_s = make_activation_envelope(
        true_state, fs, rng
    )

    # 旧实测数据的Channel1_raw基线约1000；Relax std约4.6，
    # Clench典型std约14.7，并且不同试次差异较大。
    baseline_adc = float(rng.normal(999.5, 2.0))
    relax_std_adc = float(rng.uniform(3.8, 5.5))
    clench_std_adc = float(rng.uniform(12.0, 24.0))

    relax_carrier = normalized_band_noise(
        sample_count, fs, 20.0, 80.0, rng
    )
    clench_carrier = normalized_band_noise(
        sample_count, fs, 30.0, 100.0, rng
    )
    slow_modulation = 1.0 + 0.12 * np.sin(
        2.0 * np.pi * float(rng.uniform(0.08, 0.20)) * time_s
        + float(rng.uniform(0.0, 2.0 * np.pi))
    )
    emg = (
        relax_std_adc * (1.0 - activation) * relax_carrier
        + clench_std_adc
        * activation
        * slow_modulation
        * clench_carrier
    )

    mains_hum = float(rng.uniform(0.5, 2.0)) * np.sin(
        2.0 * np.pi * 50.0 * time_s
        + float(rng.uniform(0.0, 2.0 * np.pi))
    )
    baseline_drift = float(rng.uniform(1.0, 3.5)) * np.sin(
        2.0 * np.pi * float(rng.uniform(0.15, 0.45)) * time_s
        + float(rng.uniform(0.0, 2.0 * np.pi))
    )
    motion_artifact = make_motion_artifact(
        sample_count, fs, true_events, rng
    )
    sensor_noise = rng.normal(0.0, 0.7, sample_count)

    raw_float = (
        baseline_adc
        + emg
        + mains_hum
        + baseline_drift
        + motion_artifact
        + sensor_noise
    )
    channel1_raw = np.clip(
        np.rint(raw_float),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)

    frame = pd.DataFrame(
        {
            "frame_index": np.arange(sample_count, dtype=np.int64),
            "recording_sample_index": np.arange(
                sample_count, dtype=np.int64
            ),
            "recording_elapsed_s": time_s,
            "button_label": button_label,
            "Channel1_raw": channel1_raw,
            "Channel1_display": channel1_raw.astype(np.float64)
            - baseline_adc,
            # 以下sim_*列只用于回归测试诊断，现有特征提取器不会读取它们。
            "sim_true_label": true_state,
            "sim_activation_envelope": activation,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

    mismatch_samples = int(np.count_nonzero(button_label != true_state))
    offsets_ms = [
        float(event["button_minus_true_ms"]) for event in button_events
    ]
    return {
        "split": split,
        "session_index": session_index,
        "file": path.name,
        "samples": sample_count,
        "duration_s": sample_count / fs,
        "baseline_adc": baseline_adc,
        "relax_target_std_adc": relax_std_adc,
        "clench_target_std_adc": clench_std_adc,
        "activation_rise_tau_ms": 1000.0 * rise_tau_s,
        "activation_fall_tau_ms": 1000.0 * fall_tau_s,
        "button_true_mismatch_samples": mismatch_samples,
        "button_true_mismatch_ms": 1000.0 * mismatch_samples / fs,
        "button_edge_offsets_ms": offsets_ms,
        "events": button_events,
        "clench_bursts": clench_bursts,
    }


def generate_dataset(
    data_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """生成相互独立的训练Session和验证Session。"""
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    total_sessions = args.train_sessions + args.validation_sessions
    seed_sequences = np.random.SeedSequence(args.seed).spawn(total_sessions)
    sessions: list[dict[str, Any]] = []
    sequence_index = 0

    for split, count, directory in (
        ("train", args.train_sessions, train_dir),
        ("validation", args.validation_sessions, validation_dir),
    ):
        for session_index in range(1, count + 1):
            rng = np.random.default_rng(seed_sequences[sequence_index])
            sequence_index += 1
            path = directory / (
                f"synthetic_{split}_session_{session_index:03d}.csv"
            )
            sessions.append(
                generate_one_session(
                    path=path,
                    split=split,
                    session_index=session_index,
                    duration_s=args.duration_s,
                    clench_durations_ms=args.clench_durations_ms,
                    fs=SOURCE_FS,
                    jitter_std_ms=args.button_jitter_std_ms,
                    jitter_max_ms=args.button_jitter_max_ms,
                    rng=rng,
                )
            )

    offsets = [
        offset
        for session in sessions
        for offset in session["button_edge_offsets_ms"]
    ]
    mismatch_samples = sum(
        int(session["button_true_mismatch_samples"])
        for session in sessions
    )
    if not offsets or mismatch_samples == 0:
        raise RuntimeError(
            "Simulation did not create button/true-label mismatch."
        )

    manifest: dict[str, Any] = {
        "format": "synthetic_semg_button_misalignment_v2",
        "warning": (
            "Synthetic data validates software logic only; never report its "
            "accuracy as real-system performance."
        ),
        "seed": args.seed,
        "source_sampling_rate_hz": SOURCE_FS,
        "requested_clench_durations_ms": args.clench_durations_ms,
        "button_jitter_distribution": {
            "type": "zero-mean Gaussian clipped symmetrically",
            "std_ms": args.button_jitter_std_ms,
            "absolute_max_ms": args.button_jitter_max_ms,
            "sign_definition": (
                "negative=button before true action; "
                "positive=button after true action"
            ),
        },
        "total_button_true_mismatch_samples": mismatch_samples,
        "edge_offset_summary_ms": {
            "count": len(offsets),
            "minimum": float(np.min(offsets)),
            "maximum": float(np.max(offsets)),
            "mean": float(np.mean(offsets)),
            "standard_deviation": float(np.std(offsets)),
        },
        "diagnostic_columns": {
            "button_label": "label visible to feature extractor/SVM",
            "sim_true_label": "hidden actual Relax/Clench ground truth",
            "sim_activation_envelope": "smooth physiological activation",
        },
        "sessions": sessions,
    }
    (data_dir / "synthetic_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def run_command(command: list[str], log_path: Path) -> None:
    """运行一个子流程并保存完整Terminal输出。"""
    result = subprocess.run(
        command,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="backslashreplace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nOUTPUT\n"
        + result.stdout,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}; "
            f"see {log_path.resolve()}"
        )


def exactly_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise AssertionError(
            f"Expected exactly one {description}, found {len(paths)}."
        )
    return paths[0]


def make_transition_band_mask(
    labels: np.ndarray,
    band_samples: int,
) -> np.ndarray:
    """复现 feature extractor 的按钮边沿 transition-band 标记规则。"""
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    transition_band = np.zeros(labels.size, dtype=bool)
    for edge in changes:
        start = max(0, int(edge) - band_samples)
        stop = min(labels.size, int(edge) + band_samples + 1)
        transition_band[start:stop] = True
    return transition_band


def verify_kept_windows(
    feature_csv: Path,
    source_dir: Path,
    transition_band_samples: int,
    clench_fraction_threshold: float,
) -> dict[str, int | float]:
    """确认完整候选窗口全部保留且窗口投票标签和审计列正确。"""
    features = pd.read_csv(feature_csv)
    numeric_features = features[ALL_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_features).all():
        raise AssertionError(f"{feature_csv}: features contain NaN/Inf.")

    contaminated_windows = 0
    transition_band_windows = 0
    mixed_label_windows = 0
    checked_windows = 0
    for source_name, rows in features.groupby("source_file", sort=False):
        source = pd.read_csv(
            source_dir / str(source_name),
            usecols=["button_label", "sim_true_label"],
        )
        button = source["button_label"].to_numpy(dtype=np.uint8)
        true_label = source["sim_true_label"].to_numpy(dtype=np.uint8)
        transition_band = make_transition_band_mask(
            button, transition_band_samples
        )
        expected_window_count = max(
            0,
            1 + (button.size - REFERENCE_WINDOW_SAMPLES)
            // REFERENCE_HOP_SAMPLES,
        )
        if len(rows) != expected_window_count:
            raise AssertionError(
                f"{feature_csv}: {source_name} kept {len(rows)} of "
                f"{expected_window_count} complete candidate windows."
            )
        expected_indices = np.arange(expected_window_count)
        if not np.array_equal(
            rows["window_index"].to_numpy(dtype=np.int64),
            expected_indices,
        ):
            raise AssertionError(
                f"{feature_csv}: {source_name} has missing/reordered windows."
            )

        for row in rows.itertuples(index=False):
            start = int(row.start_sample)
            stop = int(row.end_sample_exclusive)
            label_id = int(row.label_id)
            window_button = button[start:stop]
            clench_sample_count = int(np.count_nonzero(window_button))
            clench_fraction = clench_sample_count / window_button.size
            expected_label = int(
                clench_fraction > clench_fraction_threshold
            )
            contains_transition = bool(
                np.any(window_button[1:] != window_button[:-1])
            )
            intersects_band = bool(np.any(transition_band[start:stop]))
            if label_id != expected_label:
                raise AssertionError(
                    f"{feature_csv}: window vote label is incorrect."
                )
            if int(row.clench_sample_count) != clench_sample_count:
                raise AssertionError(
                    f"{feature_csv}: clench_sample_count is incorrect."
                )
            if not np.isclose(float(row.clench_fraction), clench_fraction):
                raise AssertionError(
                    f"{feature_csv}: clench_fraction is incorrect."
                )
            if bool(row.contains_label_transition) != contains_transition:
                raise AssertionError(
                    f"{feature_csv}: transition audit flag is incorrect."
                )
            if bool(row.intersects_transition_band) != intersects_band:
                raise AssertionError(
                    f"{feature_csv}: transition-band flag is incorrect."
                )
            mixed_label_windows += int(contains_transition)
            transition_band_windows += int(intersects_band)
            if not np.all(true_label[start:stop] == label_id):
                contaminated_windows += 1
            checked_windows += 1

    return {
        "checked_windows": checked_windows,
        "mixed_label_windows": mixed_label_windows,
        "transition_band_windows": transition_band_windows,
        "windows_with_any_hidden_true_label_disagreement": (
            contaminated_windows
        ),
        "hidden_true_disagreement_fraction": (
            contaminated_windows / checked_windows
            if checked_windows
            else 0.0
        ),
    }


def verify_merged_svm(
    model_path: Path,
    params_path: Path,
    validation_feature_csv: Path,
) -> int:
    """确认导出的合并权重与Python sklearn模型逐窗口判断完全一致。"""
    artifact = joblib.load(model_path)
    parameters = json.loads(params_path.read_text(encoding="utf-8"))
    feature_order = list(parameters["feature_order"])
    frame = pd.read_csv(validation_feature_csv)
    x_frame = frame[feature_order]
    x = x_frame.to_numpy(dtype=np.float64)
    weight = np.asarray(
        parameters["merged_svm_weight_raw_feature_space"],
        dtype=np.float64,
    )
    bias = float(parameters["merged_svm_bias_raw_feature_space"])
    merged_prediction = (x @ weight + bias >= 0.0).astype(np.int64)
    sklearn_prediction = artifact["model"].predict(x_frame)
    if not np.array_equal(merged_prediction, sklearn_prediction):
        raise AssertionError(
            "Merged STM32 SVM parameters disagree with sklearn predictions."
        )
    return int(merged_prediction.size)


def inspect_experiment(
    experiment_dir: Path,
    train_dir: Path,
    validation_dir: Path,
    downsample_factor: int,
    transition_band_ms: float,
    clench_fraction_threshold: float,
    minimum_balanced_accuracy: float,
) -> dict[str, Any]:
    """检查一个D对应的特征、模型、验证指标和硬件导出参数。"""
    cache_root = experiment_dir / "_feature_cache"
    train_feature_csv = exactly_one(
        list(cache_root.glob("*/train_features.csv")),
        "training feature CSV",
    )
    validation_feature_csv = exactly_one(
        list(cache_root.glob("*/validation_features.csv")),
        "validation feature CSV",
    )
    metadata_path = train_feature_csv.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    effective_fs = SOURCE_FS / downsample_factor
    expected_window = REFERENCE_WINDOW_SAMPLES // downsample_factor
    expected_hop = REFERENCE_HOP_SAMPLES // downsample_factor
    expected_high = min(REQUESTED_BPF_HIGH_HZ, 0.45 * effective_fs)
    transition_band_samples = int(
        round(transition_band_ms * SOURCE_FS / 1000.0)
    )

    assert metadata["downsampling"]["factor_D"] == downsample_factor
    assert metadata["input"]["effective_sampling_rate_hz"] == effective_fs
    assert metadata["segmentation"]["window_samples"] == expected_window
    assert metadata["segmentation"]["hop_samples"] == expected_hop
    assert metadata["fft"]["fft_length"] == expected_window
    assert metadata["segmentation"]["window_duration_ms"] == 256.0
    assert metadata["segmentation"]["hop_duration_ms"] == 128.0
    assert metadata["fft"]["frequency_resolution_hz"] == 3.90625
    assert metadata["bpf"]["total_digital_filter_order"] == 4
    assert len(metadata["bpf"]["scipy_sos_b0_b1_b2_a0_a1_a2"]) == 2
    assert metadata["bpf"]["effective_high_cutoff_hz"] == expected_high
    assert metadata["fft"]["effective_band_hz"][1] == expected_high
    assert (
        metadata["labeling"]["transition_band_source_samples_each_side"]
        == transition_band_samples
    )
    assert (
        metadata["labeling"]["clench_fraction_threshold"]
        == clench_fraction_threshold
    )
    assert (
        metadata["labeling"]["all_complete_candidate_windows_are_kept"]
        is True
    )

    if downsample_factor > 1:
        anti_alias = metadata["anti_alias_filter"]
        assert anti_alias["enabled"] is True
        sos = np.asarray(
            anti_alias["scipy_sos_b0_b1_b2_a0_a1_a2"],
            dtype=np.float64,
        )
        stopband_edge = float(anti_alias["stopband_edge_hz"])
        _, response = sosfreqz(
            sos,
            worN=[stopband_edge],
            fs=SOURCE_FS,
        )
        attenuation_db = float(
            20.0 * np.log10(max(abs(response[0]), 1e-300))
        )
        if attenuation_db > -float(
            anti_alias["minimum_stopband_attenuation_db"]
        ) + 1e-6:
            raise AssertionError(
                "Anti-alias filter misses its stopband attenuation target."
            )
    else:
        attenuation_db = None
        assert metadata["anti_alias_filter"]["enabled"] is False

    train_window_check = verify_kept_windows(
        train_feature_csv,
        train_dir,
        transition_band_samples,
        clench_fraction_threshold,
    )
    validation_window_check = verify_kept_windows(
        validation_feature_csv,
        validation_dir,
        transition_band_samples,
        clench_fraction_threshold,
    )

    summary_path = experiment_dir / "experiment_summary.csv"
    summary = pd.read_csv(summary_path)
    if len(summary) != 1 or summary.iloc[0]["status"] != "ok":
        raise AssertionError(f"Unexpected quick summary: {summary_path}")
    balanced_accuracy = float(summary.iloc[0]["balanced_accuracy"])
    if balanced_accuracy < minimum_balanced_accuracy:
        raise AssertionError(
            f"D={downsample_factor} synthetic balanced accuracy "
            f"{balanced_accuracy:.4f} is below regression floor "
            f"{minimum_balanced_accuracy:.4f}."
        )

    model_path = exactly_one(
        list(experiment_dir.glob("*/linear_svm.joblib")),
        "Linear SVM model",
    )
    params_path = model_path.with_name("linear_svm_params.json")
    merged_prediction_count = verify_merged_svm(
        model_path, params_path, validation_feature_csv
    )
    return {
        "downsample_factor": downsample_factor,
        "effective_sampling_rate_hz": effective_fs,
        "window_fft_samples": expected_window,
        "hop_samples": expected_hop,
        "effective_high_hz": expected_high,
        "anti_alias_stopband_response_db": attenuation_db,
        "balanced_accuracy": balanced_accuracy,
        "train_window_check": train_window_check,
        "validation_window_check": validation_window_check,
        "merged_svm_predictions_checked": merged_prediction_count,
        "feature_metadata": str(metadata_path.resolve()),
        "experiment_summary": str(summary_path.resolve()),
    }


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_svm_experiments.py"
    if not runner.exists():
        raise FileNotFoundError(
            f"run_svm_experiments.py must be beside this script: {runner}"
        )

    run_dir = make_unique_run_dir(args.output_root, args.seed)
    data_dir = run_dir / "data"
    print(f"Creating synthetic sessions: {data_dir.resolve()}")
    manifest = generate_dataset(data_dir, args)
    offsets = manifest["edge_offset_summary_ms"]
    print(
        "Clench duration set: "
        + ", ".join(
            f"{duration_ms:g} ms"
            for duration_ms in args.clench_durations_ms
        )
    )
    print(
        "Button edge error: "
        f"count={offsets['count']}, "
        f"min={offsets['minimum']:.1f} ms, "
        f"max={offsets['maximum']:.1f} ms, "
        f"std={offsets['standard_deviation']:.1f} ms"
    )
    print(
        "Button/true-label mismatched samples: "
        f"{manifest['total_button_true_mismatch_samples']}"
    )

    if args.generate_only:
        print("Generate-only mode: pipeline was not executed.")
        print(f"Manifest: {(data_dir / 'synthetic_manifest.json').resolve()}")
        return 0

    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    experiment_dirs: dict[int, Path] = {}
    for downsample_factor in (1, 2):
        experiment_dir = run_dir / f"experiments_d{downsample_factor}"
        experiment_dirs[downsample_factor] = experiment_dir
        print(
            f"Running quick pipeline for D={downsample_factor} "
            f"(full extraction + TD3 SVM)..."
        )
        run_command(
            [
                sys.executable,
                str(runner),
                "--train-input-dir",
                str(train_dir),
                "--validation-input-dir",
                str(validation_dir),
                "--output-dir",
                str(experiment_dir),
                "--downsample-factors",
                str(downsample_factor),
                "--transition-band-ms",
                str(args.transition_band_ms),
                "--clench-fraction-threshold",
                str(args.clench_fraction_threshold),
                "--quick",
            ],
            run_dir / "logs" / f"runner_d{downsample_factor}.log",
        )

    checks = {
        str(downsample_factor): inspect_experiment(
            experiment_dir=experiment_dirs[downsample_factor],
            train_dir=train_dir,
            validation_dir=validation_dir,
            downsample_factor=downsample_factor,
            transition_band_ms=args.transition_band_ms,
            clench_fraction_threshold=args.clench_fraction_threshold,
            minimum_balanced_accuracy=args.minimum_balanced_accuracy,
        )
        for downsample_factor in (1, 2)
    }

    # D 只改变采样点数，不改变窗口对应的原始时间区间，因此候选窗数量应相同。
    for split_key in ("train_window_check", "validation_window_check"):
        d1_count = checks["1"][split_key]["checked_windows"]
        d2_count = checks["2"][split_key]["checked_windows"]
        if d1_count != d2_count:
            raise AssertionError(
                f"D=1/D=2 complete-window count differs for {split_key}: "
                f"{d1_count} vs {d2_count}."
            )

    report = {
        "status": "PASS",
        "warning": manifest["warning"],
        "run_directory": str(run_dir.resolve()),
        "synthetic_manifest": str(
            (data_dir / "synthetic_manifest.json").resolve()
        ),
        "simulation": {
            "seed": args.seed,
            "train_sessions": args.train_sessions,
            "validation_sessions": args.validation_sessions,
            "duration_s_each": args.duration_s,
            "clench_durations_ms": args.clench_durations_ms,
            "transition_band_ms": args.transition_band_ms,
            "clench_fraction_threshold": args.clench_fraction_threshold,
            "button_jitter_std_ms": args.button_jitter_std_ms,
            "button_jitter_max_ms": args.button_jitter_max_ms,
            "edge_offset_summary_ms": offsets,
            "button_true_mismatch_samples": (
                manifest["total_button_true_mismatch_samples"]
            ),
        },
        "checks": checks,
    }
    report_path = run_dir / "quick_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPASS: synthetic quick validation completed.")
    for downsample_factor in (1, 2):
        result = checks[str(downsample_factor)]
        contamination = result["validation_window_check"][
            "hidden_true_disagreement_fraction"
        ]
        print(
            f"D={downsample_factor}: "
            f"balanced_accuracy={result['balanced_accuracy']:.4f}, "
            f"NFFT={result['window_fft_samples']}, "
            f"validation hidden-label contamination="
            f"{contamination:.2%}"
        )
    print(f"Report: {report_path.resolve()}")
    print(f"Synthetic CSVs: {data_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
