#!/usr/bin/env python3
"""从连续 sEMG Session CSV 中提取稳定状态窗口特征。

输入目录中的每个 ``*.csv`` 都代表一段连续采集 Session，并且必须包含：

* ``Channel1_raw``：单通道原始 sEMG ADC 数据；
* ``button_label``：PC GUI 虚拟按钮状态，0=Relax，1=Clench。

本脚本不再根据文件名推断标签。它先对整段 Session 连续执行可部署到
STM32 的因果 Butterworth 带通滤波，再按固定长度滑动窗口。按钮上升沿和
下降沿前后的保护区，以及包含混合按钮标签的窗口，都会被丢弃。

默认设置：

* source sampling rate = 500 Hz，digital downsampling factor D = 1；
* D > 1 时使用因果 Chebyshev-I anti-alias LPF（1 dB / 40 dB）；
* causal total-4th-order Butterworth BPF：f_low = 20 Hz，
  effective_f_high = min(requested_f_high, 0.45 * effective_fs)；
* transition guard = 每个按钮边沿前后各 50 ms；
* reference window = 128 source samples，overlap = 50%；
* 降采样后 window/FFT points = 128/D，hop = 64/D，因此窗口时长和
  更新周期不变（不代表滤波群延迟或实际处理时间完全相同）；
* 每个有效窗口独立生成一个标签，不进行多数投票；
* 同时支持时域特征和基于 Hann + RFFT 的频域特征。

运行示例：

    python emg_feature_extractor.py --input-dir data_collection/state_sessions
    python emg_feature_extractor.py --downsample-factor 2
    python emg_feature_extractor.py --features mav rms wl
    python emg_feature_extractor.py --features mnf mdf pkf bandpower
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, cheb1ord, cheby1, sosfilt


# 数字标签会直接写入 label bin，后续训练和 STM32 验证都使用相同映射。
LABEL_TO_NAME = {0: "relax", 1: "clench"}

TIME_FEATURES = ("mav", "rms", "wl", "var", "zc", "ssc", "wamp")
FREQUENCY_FEATURES = ("mnf", "mdf", "pkf", "bandpower")
AVAILABLE_FEATURES = TIME_FEATURES + FREQUENCY_FEATURES


def design_bandpass(
    fs: float,
    low_hz: float,
    high_hz: float,
    total_order: int,
) -> np.ndarray:
    """设计总阶数明确的 Butterworth BPF，返回 SciPy SOS 系数。

    SciPy 的 band-pass ``butter(N, ...)`` 会产生总阶数为 ``2*N`` 的
    滤波器。因此，为了让命令行中的 ``--filter-order 4`` 真正表示
    “总四阶 BPF”，这里传给 SciPy 的 prototype order 是 2。
    """
    nyquist_hz = fs / 2.0
    if not 0.0 < low_hz < high_hz < nyquist_hz:
        raise ValueError(
            "BPF cutoffs must satisfy "
            f"0 < low < high < Nyquist ({nyquist_hz:g} Hz); "
            f"got {low_hz:g}--{high_hz:g} Hz."
        )
    if total_order < 2 or total_order % 2:
        raise ValueError(
            "Band-pass total order must be an even integer >= 2."
        )

    prototype_order = total_order // 2
    return butter(
        prototype_order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )


def design_antialias(
    source_fs: float,
    effective_fs: float,
    passband_edge_hz: float,
    passband_ripple_db: float,
    stopband_attenuation_db: float,
) -> tuple[np.ndarray, int, float | None]:
    """设计抽取前的因果 Chebyshev-I 抗混叠低通滤波器。

    passband edge 使用需要保留的BPF上限；stopband edge 使用降采样后的
    Nyquist frequency。由明确的通带纹波/阻带衰减指标自动求最低阶数，
    避免把衰减不足的四阶BPF误当成完整的anti-aliasing filter。
    D=1时不需要抽取，因此返回空SOS。
    """
    if effective_fs == source_fs:
        return np.empty((0, 6), dtype=np.float64), 0, None

    source_nyquist_hz = source_fs / 2.0
    stopband_edge_hz = effective_fs / 2.0
    if not (
        0.0
        < passband_edge_hz
        < stopband_edge_hz
        < source_nyquist_hz
    ):
        raise ValueError(
            "Anti-alias edges must satisfy 0 < passband < effective "
            "Nyquist < source Nyquist."
        )

    order, critical_hz = cheb1ord(
        wp=passband_edge_hz,
        ws=stopband_edge_hz,
        gpass=passband_ripple_db,
        gstop=stopband_attenuation_db,
        fs=source_fs,
    )
    sos = cheby1(
        order,
        passband_ripple_db,
        critical_hz,
        btype="lowpass",
        fs=source_fs,
        output="sos",
    )
    return sos, int(order), float(critical_hz)


def causal_bandpass(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """对一整段连续 Session 执行一次因果 SOS/Biquad 滤波。

    滤波器状态只在 Session 开始时清零，随后跨越所有滑动窗口持续保留。
    这与 STM32 对连续采样流（D>1时为抽取后的流）逐点执行 Biquad 的方式
    一致。禁止逐窗口重置状态，也不使用需要未来数据的 ``filtfilt``。
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 2:
        raise ValueError("A session must contain at least two samples.")
    if not np.isfinite(x).all():
        raise ValueError("Channel1_raw contains NaN or infinite values.")

    # 未传入 zi 时，sosfilt 使用全零初始状态；STM32 启动时也应清零状态数组。
    return sosfilt(sos, x)


def make_transition_invalid_mask(
    labels: np.ndarray,
    guard_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据按钮上升/下降沿生成过渡保护区 invalid mask。

    边沿索引 ``i`` 表示 ``label[i-1] != label[i]``，即新按钮状态从采样点
    ``i`` 开始。默认 500 Hz、50 ms 时，guard=25 samples；索引距离边沿
    不超过25个采样点的位置都会标记为无效。
    """
    labels = np.asarray(labels, dtype=np.uint8)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    rising_edges = changes[
        (labels[changes - 1] == 0) & (labels[changes] == 1)
    ]
    falling_edges = changes[
        (labels[changes - 1] == 1) & (labels[changes] == 0)
    ]

    invalid = np.zeros(labels.size, dtype=bool)
    if guard_samples > 0:
        for edge in changes:
            # stop 为开区间，因此 +1 后会包含 edge+guard_samples。
            start = max(0, int(edge) - guard_samples)
            stop = min(labels.size, int(edge) + guard_samples + 1)
            invalid[start:stop] = True

    return invalid, rising_edges, falling_edges


def threshold_crossings(values: np.ndarray, threshold: float) -> int:
    """计算带幅值阈值的 Zero Crossings (ZC)。"""
    left = values[:-1]
    right = values[1:]
    sign_changed = ((left >= 0.0) & (right < 0.0)) | (
        (left < 0.0) & (right >= 0.0)
    )
    large_enough = np.abs(right - left) >= threshold
    return int(np.count_nonzero(sign_changed & large_enough))


def fft_band_features(
    values: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> dict[str, float]:
    """用 symmetric Hann window + one-sided RFFT 计算基础频域特征。

    Power spectral density (PSD) 使用 one-sided periodogram 定义：

    ``PSD[k] = |RFFT(x * Hann)[k]|^2 / (fs * sum(Hann^2))``

    除 DC 和 Nyquist 外的正频率bin乘2。Band Power 是指定频带内 PSD
    乘以 ``df`` 后求和；MNF/MDF/PKF 只使用同一频带内的功率。
    """
    x = np.asarray(values, dtype=np.float64)
    n_samples = x.size
    hann = np.hanning(n_samples)
    spectrum = np.fft.rfft(x * hann, n=n_samples)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    window_energy = float(np.sum(hann * hann))
    if window_energy <= 0.0:
        raise ValueError("FFT window energy is zero.")

    psd = (np.abs(spectrum) ** 2) / (fs * window_energy)
    if n_samples % 2 == 0:
        # 偶数点RFFT包含Nyquist：DC与Nyquist不翻倍，其余正频率翻倍。
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    band_frequencies = frequencies[band_mask]
    band_psd = psd[band_mask]
    if band_frequencies.size == 0:
        raise ValueError(
            f"No FFT bin lies inside {low_hz:g}--{high_hz:g} Hz."
        )

    total_power_density = float(np.sum(band_psd))
    df_hz = fs / n_samples
    bandpower = float(total_power_density * df_hz)

    # 全零或极低能量窗口没有有意义的频率中心，此时统一返回0，避免NaN。
    if total_power_density <= np.finfo(np.float64).tiny:
        return {
            "mnf": 0.0,
            "mdf": 0.0,
            "pkf": 0.0,
            "bandpower": bandpower,
        }

    mnf = float(
        np.sum(band_frequencies * band_psd) / total_power_density
    )
    half_power = 0.5 * total_power_density
    median_index = int(
        np.searchsorted(np.cumsum(band_psd), half_power, side="left")
    )
    # 防止极端浮点舍入使索引恰好落在数组末尾之外。
    median_index = min(median_index, band_frequencies.size - 1)
    mdf = float(band_frequencies[median_index])
    pkf = float(band_frequencies[int(np.argmax(band_psd))])

    return {
        "mnf": mnf,
        "mdf": mdf,
        "pkf": pkf,
        "bandpower": bandpower,
    }


def extract_features(
    window: np.ndarray,
    names: list[str],
    threshold: float,
    fs: float,
    fft_low_hz: float,
    fft_high_hz: float,
) -> dict[str, float]:
    """从一个稳定状态窗口提取用户指定的时域/频域特征。"""
    x = np.asarray(window, dtype=np.float64)
    if x.size < 2:
        raise ValueError("A feature window must contain at least two samples.")

    dx = np.diff(x)
    result: dict[str, float] = {}

    # FFT成本远高于简单时域运算；只有特征组合包含频域特征时才真正执行。
    frequency_result: dict[str, float] | None = None
    if any(name in FREQUENCY_FEATURES for name in names):
        frequency_result = fft_band_features(
            x, fs=fs, low_hz=fft_low_hz, high_hz=fft_high_hz
        )

    for name in names:
        if name == "mav":
            # Mean Absolute Value：平均绝对幅值。
            result[name] = float(np.mean(np.abs(x)))
        elif name == "rms":
            # Root Mean Square：信号均方根幅值。
            result[name] = float(np.sqrt(np.mean(x * x)))
        elif name == "wl":
            # Waveform Length：相邻采样点绝对差之和。
            result[name] = float(np.sum(np.abs(dx)))
        elif name == "var":
            # Population Variance：除以N，便于STM32使用固定窗口长度计算。
            mean_value = float(np.mean(x))
            result[name] = float(np.mean((x - mean_value) ** 2))
        elif name == "zc":
            result[name] = float(threshold_crossings(x, threshold))
        elif name == "ssc":
            # 中心点相对左右两点同时形成有效转折时计为一次SSC。
            left_delta = x[1:-1] - x[:-2]
            right_delta = x[1:-1] - x[2:]
            is_turning_point = left_delta * right_delta > 0.0
            exceeds_threshold = (
                (np.abs(left_delta) >= threshold)
                & (np.abs(right_delta) >= threshold)
            )
            result[name] = float(
                np.count_nonzero(is_turning_point & exceeds_threshold)
            )
        elif name == "wamp":
            # Willison Amplitude：相邻采样差大于等于阈值的次数。
            result[name] = float(
                np.count_nonzero(np.abs(dx) >= threshold)
            )
        elif name in FREQUENCY_FEATURES:
            assert frequency_result is not None
            result[name] = frequency_result[name]
        else:
            raise ValueError(f"Unsupported feature: {name}")

    return result


def read_session_csv(
    path: Path,
    channel: str,
    label_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """读取并严格验证连续 Session 的信号和逐采样按钮标签。"""
    frame = pd.read_csv(path)
    missing = [
        column
        for column in (channel, label_column)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"{path}: missing required column(s): {', '.join(missing)}"
        )

    try:
        raw = pd.to_numeric(frame[channel], errors="raise").to_numpy(
            dtype=np.float64
        )
        label_values = pd.to_numeric(
            frame[label_column], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: signal and button labels must be numeric."
        ) from exc

    if raw.size != label_values.size or raw.size == 0:
        raise ValueError(f"{path}: signal/label columns are empty or mismatched.")
    if not np.isfinite(raw).all() or not np.isfinite(label_values).all():
        raise ValueError(f"{path}: signal/label contains NaN or infinity.")
    if not np.all(np.isin(label_values, (0.0, 1.0))):
        invalid_values = np.unique(
            label_values[~np.isin(label_values, (0.0, 1.0))]
        )
        raise ValueError(
            f"{path}: button_label must contain only 0/1; "
            f"found {invalid_values.tolist()}."
        )

    return raw, label_values.astype(np.uint8)


def process_session(
    path: Path,
    args: argparse.Namespace,
    anti_alias_sos: np.ndarray,
    bpf_sos: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """连续滤波一个Session，再生成所有有效稳定状态窗口。"""
    raw, button_labels = read_session_csv(
        path, args.channel, args.label_column
    )
    # D>1时必须先在原始采样率下执行有明确阻带指标的anti-alias低通，
    # 再每D点保留1点；禁止先抽取再滤波，否则高频成分会不可逆地折叠。
    if anti_alias_sos.size:
        anti_aliased_source = sosfilt(anti_alias_sos, raw)
    else:
        anti_aliased_source = raw
    usable_source_samples = (raw.size // args.downsample_factor) * (
        args.downsample_factor
    )
    downsampled = anti_aliased_source[
        :usable_source_samples:args.downsample_factor
    ]
    # BPF在有效采样率下连续运行一次，滤波器状态跨越全部滑动窗口。
    filtered = causal_bandpass(downsampled, bpf_sos)

    invalid_mask, rising_edges, falling_edges = (
        make_transition_invalid_mask(
            button_labels, args.guard_samples
        )
    )

    rows: list[dict[str, Any]] = []
    discarded_guard = 0
    discarded_mixed = 0
    candidate_count = 0

    for candidate_index, start in enumerate(
        range(0, len(filtered) - args.window_samples + 1, args.hop_samples)
    ):
        candidate_count += 1
        end = start + args.window_samples
        source_start = start * args.downsample_factor
        source_end = end * args.downsample_factor

        # 标签和±50 ms保护区始终在原始采样轴上检查。这样D改变以后仍然严格使用
        # 相同的真实时间范围，而不是受到降采样后边沿量化误差的影响。
        if np.any(invalid_mask[source_start:source_end]):
            discarded_guard += 1
            continue

        window_labels = button_labels[source_start:source_end]
        label_id = int(window_labels[0])
        # 即使guard设为0，也只接受整窗全部为同一个按钮状态的稳定窗口。
        if not np.all(window_labels == label_id):
            discarded_mixed += 1
            continue

        features = extract_features(
            filtered[start:end],
            names=args.features,
            threshold=args.threshold,
            fs=args.fs,
            fft_low_hz=args.fft_low_hz,
            fft_high_hz=args.effective_fft_high_hz,
        )
        rows.append(
            {
                "source_file": path.name,
                "session_id": path.stem,
                "window_index": candidate_index,
                # start/end_sample继续表示原始ADC采样索引，便于回查原始CSV。
                "start_sample": source_start,
                "end_sample_exclusive": source_end,
                "downsampled_start_sample": start,
                "downsampled_end_sample_exclusive": end,
                "start_time_s": source_start / args.source_fs,
                "end_time_s": source_end / args.source_fs,
                "label": LABEL_TO_NAME[label_id],
                "label_id": label_id,
                **features,
            }
        )

    stats: dict[str, Any] = {
        "source_file": path.name,
        "source_samples": int(raw.size),
        "downsampled_samples": int(filtered.size),
        "unused_source_tail_samples": int(raw.size - usable_source_samples),
        "duration_s": float(raw.size / args.source_fs),
        "rising_edges": int(rising_edges.size),
        "falling_edges": int(falling_edges.size),
        "candidate_windows": candidate_count,
        "valid_windows": len(rows),
        "discarded_guard": discarded_guard,
        "discarded_mixed_label": discarded_mixed,
    }
    return rows, stats


def feature_definitions(threshold: float) -> dict[str, str]:
    """返回写入JSON的精确特征定义，便于STM32端逐项复现。"""
    return {
        "mav": "mean(abs(x))",
        "rms": "sqrt(mean(x^2))",
        "wl": "sum(abs(x[n]-x[n-1]))",
        "var": "mean((x-mean(x))^2), population variance (divide by N)",
        "zc": (
            "count sign changes with abs(x[n]-x[n-1]) >= "
            f"{threshold:g}"
        ),
        "ssc": (
            "count turning points where both adjacent slope magnitudes "
            f">= {threshold:g}"
        ),
        "wamp": f"count abs(x[n]-x[n-1]) >= {threshold:g}",
        "mnf": "sum(f[k]*PSD[k])/sum(PSD[k]) inside FFT band",
        "mdf": "first FFT-bin frequency reaching 50% cumulative band PSD",
        "pkf": "frequency of maximum PSD bin inside FFT band",
        "bandpower": "sum(one-sided PSD bins inside FFT band) * df",
    }


def sos_to_cmsis_df2t(sos: np.ndarray) -> list[list[float]]:
    """把SciPy SOS转换为CMSIS-DSP DF2T使用的系数顺序。"""
    # SciPy每行是[b0,b1,b2,a0,a1,a2]，其中a0已经归一化为1；
    # CMSIS-DSP DF2T每级使用[b0,b1,b2,-a1,-a2]。
    return [
        [
            float(section[0]),
            float(section[1]),
            float(section[2]),
            float(-section[4]),
            float(-section[5]),
        ]
        for section in sos
    ]


def save_outputs(
    result: pd.DataFrame,
    session_stats: list[dict[str, Any]],
    anti_alias_sos: np.ndarray,
    bpf_sos: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path]:
    """保存CSV、float32特征bin、uint8标签bin和JSON元数据。"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    feature_path = args.binary_output or args.output.with_suffix(".bin")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    label_path = feature_path.with_name(f"{feature_path.stem}_labels.bin")
    metadata_path = feature_path.with_suffix(".json")

    # 固定row-major顺序：[window0全部特征, window1全部特征, ...]。
    feature_matrix = np.asarray(
        result[args.features].to_numpy(),
        dtype="<f4",
        order="C",
    )
    feature_matrix.tofile(feature_path)
    labels = result["label_id"].to_numpy(dtype=np.uint8)
    labels.tofile(label_path)

    fft_frequencies = np.fft.rfftfreq(
        args.window_samples, d=1.0 / args.fs
    )
    fft_indices = np.flatnonzero(
        (fft_frequencies >= args.fft_low_hz)
        & (fft_frequencies <= args.effective_fft_high_hz)
    )

    anti_alias_cmsis_df2t = sos_to_cmsis_df2t(anti_alias_sos)
    bpf_cmsis_df2t = sos_to_cmsis_df2t(bpf_sos)

    counts = result.groupby("label_id").size().to_dict()
    metadata: dict[str, Any] = {
        "format": "semg_state_window_features_v2",
        "csv_file": args.output.name,
        "feature_file": feature_path.name,
        "feature_dtype": "float32",
        "byte_order": "little-endian",
        "layout": "row-major",
        "feature_shape": [
            int(feature_matrix.shape[0]),
            int(feature_matrix.shape[1]),
        ],
        "feature_order": args.features,
        "bytes_per_feature": 4,
        "bytes_per_window": int(feature_matrix.shape[1] * 4),
        "label_file": label_path.name,
        "label_dtype": "uint8",
        "label_shape": [int(labels.shape[0])],
        "label_mapping": {"0": "relax", "1": "clench"},
        "classification": {
            "level": "stable_state_sliding_window",
            "one_prediction_per_valid_window": True,
            "majority_vote": False,
        },
        "hardware_timing_contract": {
            "preprocessing_total_us": (
                "causal anti-alias LPF for new source-rate samples when D>1, "
                "D-to-1 decimation, causal BPF for new effective-rate "
                "samples, plus all selected time/frequency feature "
                "extraction including Hann and FFT"
            ),
            "svm_inference_us": (
                "feature scaling/merged linear-SVM score and binary decision"
            ),
            "total_pipeline_us": (
                "preprocessing_total_us + svm_inference_us"
            ),
            "excluded": (
                "ADC/window acquisition wait, UART, GUI and display refresh"
            ),
        },
        "input": {
            "source_sampling_rate_hz": args.source_fs,
            "effective_sampling_rate_hz": args.fs,
            "signal_column": args.channel,
            "label_column": args.label_column,
            "label_source": "PC GUI virtual button recorded per sample",
            "one_csv_is_one_continuous_session": True,
        },
        "downsampling": {
            "factor_D": args.downsample_factor,
            "order": (
                "causal anti-alias LPF, keep every D-th sample, causal BPF"
            ),
            "source_sampling_rate_hz": args.source_fs,
            "effective_sampling_rate_hz": args.fs,
            "nyquist_after_downsampling_hz": args.fs / 2.0,
            "response_time_policy": (
                "reference window and hop are both divided by D; this "
                "preserves nominal window duration and update cadence, "
                "not filter group delay or measured processing time"
            ),
        },
        "anti_alias_filter": {
            "enabled": bool(anti_alias_sos.size),
            "type": (
                "Chebyshev type I low-pass" if anti_alias_sos.size else None
            ),
            "causal": True,
            "source_sampling_rate_hz": args.source_fs,
            "passband_edge_hz": (
                args.effective_high_hz if anti_alias_sos.size else None
            ),
            "stopband_edge_hz": (
                args.fs / 2.0 if anti_alias_sos.size else None
            ),
            "maximum_passband_ripple_db": (
                args.aa_passband_ripple_db
                if anti_alias_sos.size
                else None
            ),
            "minimum_stopband_attenuation_db": (
                args.aa_stopband_attenuation_db
                if anti_alias_sos.size
                else None
            ),
            "digital_filter_order": args.aa_filter_order,
            "scipy_critical_frequency_hz": args.aa_critical_hz,
            "implementation": "SOS/biquad cascade, direct-form II transposed",
            "state_rule": (
                "zero-initialize once at each CSV session start; preserve "
                "state continuously across the source stream"
            ),
            "scipy_sos_b0_b1_b2_a0_a1_a2": anti_alias_sos.tolist(),
            "cmsis_df2t_b0_b1_b2_neg_a1_neg_a2": (
                anti_alias_cmsis_df2t
            ),
        },
        "segmentation": {
            "reference_window_samples_at_source_rate": (
                args.reference_window_samples
            ),
            "window_samples": args.window_samples,
            "window_duration_ms": 1000.0 * args.window_samples / args.fs,
            "requested_overlap_fraction": args.overlap,
            "source_hop_samples": args.source_hop_samples,
            "hop_samples": args.hop_samples,
            "hop_duration_ms": 1000.0 * args.hop_samples / args.fs,
            "actual_overlap_fraction": (
                1.0 - args.hop_samples / args.window_samples
            ),
            "window_start_rule": (
                "starts at sample 0, then advances by hop_samples"
            ),
        },
        "labeling": {
            "stable_window_rule": (
                "all button_label samples in a kept window must be identical"
            ),
            "transition_guard_ms_each_side": args.guard_ms,
            "transition_guard_source_samples_each_side": args.guard_samples,
            "guard_and_label_checks_use_source_sample_axis": True,
            "edge_definition": (
                "edge i means button_label[i-1] != button_label[i]"
            ),
            "invalid_interval_definition": (
                "for guard_samples > 0, sample indices from "
                "edge-guard_samples through edge+guard_samples inclusive"
            ),
            "discard_if_window_intersects_invalid_mask": True,
        },
        "bpf": {
            "type": "Butterworth band-pass",
            "causal": True,
            "zero_phase": False,
            "sampling_rate_hz": args.fs,
            "low_cutoff_hz": args.low_hz,
            "requested_high_cutoff_hz": args.requested_high_hz,
            "effective_high_cutoff_hz": args.effective_high_hz,
            "high_cutoff_hz": args.effective_high_hz,
            "effective_high_rule": (
                "min(requested_high_cutoff_hz, "
                "0.45 * effective_sampling_rate_hz)"
            ),
            "total_digital_filter_order": args.filter_order,
            "scipy_bandpass_prototype_order": args.filter_order // 2,
            "implementation": "SOS/biquad cascade, direct-form II transposed",
            "state_rule": (
                "zero-initialize once at each CSV session start; preserve "
                "state continuously across all windows"
            ),
            "scipy_sos_b0_b1_b2_a0_a1_a2": bpf_sos.tolist(),
            "cmsis_df2t_b0_b1_b2_neg_a1_neg_a2": bpf_cmsis_df2t,
        },
        "fft": {
            "fft_length": args.window_samples,
            "input": "decimated causal-BPF output; no rectification",
            "window": (
                "symmetric Hann: 0.5-0.5*cos(2*pi*n/(N-1))"
            ),
            "transform": "one-sided real FFT (RFFT)",
            "frequency_resolution_hz": args.fs / args.window_samples,
            "requested_band_hz": [
                args.fft_low_hz,
                args.requested_fft_high_hz,
            ],
            "effective_band_hz": [
                args.fft_low_hz,
                args.effective_fft_high_hz,
            ],
            "effective_high_rule": (
                "min(requested_fft_high_hz, "
                "0.45 * effective_sampling_rate_hz)"
            ),
            "included_bin_indices": fft_indices.tolist(),
            "included_bin_frequencies_hz": (
                fft_frequencies[fft_indices].tolist()
            ),
            "psd_definition": (
                "|RFFT(x*Hann)|^2/(fs*sum(Hann^2)); positive bins doubled "
                "except DC and Nyquist"
            ),
            "bandpower_definition": "sum(in-band one-sided PSD) * df",
        },
        "features": {
            "selected": args.features,
            "definitions": feature_definitions(args.threshold),
            "zc_ssc_wamp_threshold_filtered_adc_counts": args.threshold,
        },
        "counts": {
            "sessions": len(session_stats),
            "candidate_windows": int(
                sum(item["candidate_windows"] for item in session_stats)
            ),
            "valid_windows": int(len(result)),
            "discarded_guard": int(
                sum(item["discarded_guard"] for item in session_stats)
            ),
            "discarded_mixed_label": int(
                sum(
                    item["discarded_mixed_label"]
                    for item in session_stats
                )
            ),
            "relax_windows": int(counts.get(0, 0)),
            "clench_windows": int(counts.get(1, 0)),
        },
        "session_statistics": session_stats,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return args.output, feature_path, label_path, metadata_path


def parse_args() -> argparse.Namespace:
    """定义并检查全部可调预处理、窗口和特征参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Extract stable Relax/Clench window features from continuous "
            "CSV sessions with sample-level PC-GUI button labels."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_collection/state_sessions"),
        help="Directory containing continuous-session *.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("emg_state_features.csv"),
    )
    parser.add_argument(
        "--binary-output",
        type=Path,
        help=(
            "Float32 feature bin path. Default: CSV output with .bin suffix."
        ),
    )
    parser.add_argument("--channel", default="Channel1_raw")
    parser.add_argument("--label-column", default="button_label")

    # --sampling-rate表示原始ADC采样率；BPF后再按整数D执行抽取。
    parser.add_argument(
        "--sampling-rate", "--fs",
        dest="source_fs",
        type=float,
        default=500.0,
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=1,
        help=(
            "Integer decimation factor D. Effective fs, window/FFT length "
            "and hop are divided by D to preserve nominal window duration "
            "and update cadence."
        ),
    )
    parser.add_argument(
        "--aa-passband-ripple-db",
        type=float,
        default=1.0,
        help="Maximum Chebyshev-I anti-alias passband ripple for D>1.",
    )
    parser.add_argument(
        "--aa-stopband-attenuation-db",
        type=float,
        default=40.0,
        help="Minimum anti-alias attenuation at effective Nyquist for D>1.",
    )

    # 可部署的因果Butterworth BPF参数。
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)
    parser.add_argument(
        "--filter-order",
        type=int,
        default=4,
        help="Total digital BPF order; must be even.",
    )

    # PC GUI按钮每个上升/下降沿前后各丢弃50 ms。
    parser.add_argument(
        "--edge-guard-ms", "--guard-ms",
        dest="guard_ms",
        type=float,
        default=50.0,
    )

    # 这里始终输入原始采样率下的参考点数。实际窗口/FFT点数会自动除以D。
    parser.add_argument(
        "--window-samples",
        dest="reference_window_samples",
        type=int,
        default=128,
        help="Reference window length at the source sampling rate.",
    )
    parser.add_argument("--overlap", type=float, default=0.5)

    # FFT频带可与BPF相同，也允许后续作为实验参数单独调整。
    parser.add_argument("--fft-low-hz", type=float, default=20.0)
    parser.add_argument("--fft-high-hz", type=float, default=100.0)

    parser.add_argument(
        "--features",
        nargs="+",
        choices=AVAILABLE_FEATURES,
        default=list(AVAILABLE_FEATURES),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Filtered-ADC-count threshold used by ZC, SSC and WAMP.",
    )
    args = parser.parse_args()

    if args.source_fs <= 0.0:
        parser.error("--sampling-rate/--fs must be positive.")
    if args.downsample_factor < 1:
        parser.error("--downsample-factor must be a positive integer.")
    if args.reference_window_samples < 2:
        parser.error("--window-samples must be >= 2.")
    if not 0.0 <= args.overlap < 1.0:
        parser.error("--overlap must satisfy 0 <= overlap < 1.")
    if args.guard_ms < 0.0:
        parser.error("--guard-ms cannot be negative.")
    if args.threshold < 0.0:
        parser.error("--threshold cannot be negative.")
    if args.aa_passband_ripple_db <= 0.0:
        parser.error("--aa-passband-ripple-db must be positive.")
    if (
        args.aa_stopband_attenuation_db
        <= args.aa_passband_ripple_db
    ):
        parser.error(
            "--aa-stopband-attenuation-db must exceed "
            "--aa-passband-ripple-db."
        )
    if args.filter_order < 2 or args.filter_order % 2:
        parser.error("--filter-order must be an even integer >= 2.")
    if len(set(args.features)) != len(args.features):
        parser.error("--features cannot contain duplicate names.")

    effective_fs = args.source_fs / args.downsample_factor
    safe_high_limit_hz = 0.45 * effective_fs
    args.requested_high_hz = args.high_hz
    args.effective_high_hz = min(
        args.requested_high_hz, safe_high_limit_hz
    )
    if not 0.0 < args.low_hz < args.effective_high_hz:
        parser.error(
            "BPF cutoffs must satisfy 0 < low-hz < "
            "min(high-hz, 0.45 * effective sampling rate); "
            f"effective high is {args.effective_high_hz:g} Hz."
        )

    # FFT上限也不能越过0.45*effective_fs；若用户把FFT/BPF设成同一频带，
    # 两者会得到完全相同的effective high。仍允许FFT请求上限低于BPF上限。
    args.requested_fft_high_hz = args.fft_high_hz
    args.effective_fft_high_hz = min(
        args.requested_fft_high_hz, safe_high_limit_hz
    )
    if (
        not 0.0
        <= args.fft_low_hz
        < args.effective_fft_high_hz
    ):
        parser.error(
            "FFT band must satisfy 0 <= fft-low-hz < "
            "min(fft-high-hz, 0.45 * effective sampling rate)."
        )

    reference_hop_samples = int(
        round(args.reference_window_samples * (1.0 - args.overlap))
    )
    if reference_hop_samples < 1:
        parser.error(
            "Window/overlap produces hop_samples < 1; reduce overlap."
        )
    if args.reference_window_samples % args.downsample_factor:
        parser.error(
            "--window-samples must be divisible by --downsample-factor "
            "so window duration remains exactly unchanged."
        )
    if reference_hop_samples % args.downsample_factor:
        parser.error(
            "The reference hop must be divisible by --downsample-factor "
            "so update period remains exactly unchanged."
        )

    # ±50 ms标签保护区在原始采样轴计算，因此不会随D产生额外量化误差。
    args.guard_samples = int(
        round(args.guard_ms * args.source_fs / 1000.0)
    )
    args.source_hop_samples = reference_hop_samples
    args.fs = effective_fs
    args.window_samples = (
        args.reference_window_samples // args.downsample_factor
    )
    args.hop_samples = (
        args.source_hop_samples // args.downsample_factor
    )
    if args.window_samples < 2:
        parser.error(
            "Downsampling leaves fewer than two samples per feature window."
        )
    return args


def main() -> int:
    """查找Session CSV、连续处理、切窗并保存四种输出。"""
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {args.input_dir}")

    output_resolved = args.output.resolve()
    paths = sorted(
        path
        for path in args.input_dir.glob("*.csv")
        if path.resolve() != output_resolved
    )
    if not paths:
        raise FileNotFoundError(f"No *.csv files found in {args.input_dir}")

    anti_alias_sos, aa_filter_order, aa_critical_hz = design_antialias(
        source_fs=args.source_fs,
        effective_fs=args.fs,
        passband_edge_hz=args.effective_high_hz,
        passband_ripple_db=args.aa_passband_ripple_db,
        stopband_attenuation_db=args.aa_stopband_attenuation_db,
    )
    args.aa_filter_order = aa_filter_order
    args.aa_critical_hz = aa_critical_hz
    bpf_sos = design_bandpass(
        fs=args.fs,
        low_hz=args.low_hz,
        high_hz=args.effective_high_hz,
        total_order=args.filter_order,
    )

    all_rows: list[dict[str, Any]] = []
    all_stats: list[dict[str, Any]] = []
    for path in paths:
        rows, stats = process_session(
            path, args, anti_alias_sos, bpf_sos
        )
        all_rows.extend(rows)
        all_stats.append(stats)
        print(
            f"{path.name}: candidates={stats['candidate_windows']}, "
            f"kept={stats['valid_windows']}, "
            f"guard-discarded={stats['discarded_guard']}, "
            f"mixed-discarded={stats['discarded_mixed_label']}"
        )

    if not all_rows:
        raise ValueError(
            "No valid stable-state windows remain. Check button_label, "
            "session length, guard-ms, window-samples and overlap."
        )

    result = pd.DataFrame(all_rows)
    csv_path, feature_path, label_path, metadata_path = save_outputs(
        result, all_stats, anti_alias_sos, bpf_sos, args
    )

    class_counts = result.groupby("label").size().to_dict()
    candidate_total = sum(
        item["candidate_windows"] for item in all_stats
    )
    print(
        f"Processed {len(paths)} sessions: "
        f"{len(result)}/{candidate_total} valid windows"
    )
    print(f"Class counts: {class_counts}")
    print(
        f"Downsampling D={args.downsample_factor}: "
        f"fs={args.source_fs:g}->{args.fs:g} Hz, "
        f"window/FFT={args.reference_window_samples}"
        f"->{args.window_samples} samples, "
        f"hop={args.source_hop_samples}->{args.hop_samples} samples"
    )
    if anti_alias_sos.size:
        print(
            f"Anti-alias LPF: Chebyshev-I order={args.aa_filter_order}, "
            f"passband<={args.effective_high_hz:g} Hz/"
            f"{args.aa_passband_ripple_db:g} dB, "
            f"stopband>={args.fs / 2.0:g} Hz/"
            f"{args.aa_stopband_attenuation_db:g} dB"
        )
    else:
        print("Anti-alias LPF: disabled (D=1, no decimation)")
    print(
        f"BPF: total order={args.filter_order}, low={args.low_hz:g} Hz, "
        f"requested high={args.requested_high_hz:g} Hz, "
        f"effective high={args.effective_high_hz:g} Hz"
    )
    print(
        f"Timing: window={1000.0 * args.window_samples / args.fs:g} ms, "
        f"update={1000.0 * args.hop_samples / args.fs:g} ms, "
        f"guard_each_side={args.guard_samples} source samples"
    )
    print(f"Features: {args.features}")
    print(f"Saved CSV: {csv_path.resolve()}")
    print(f"Saved feature binary: {feature_path.resolve()}")
    print(f"Saved label binary: {label_path.resolve()}")
    print(f"Saved metadata: {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
