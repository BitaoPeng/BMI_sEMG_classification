#!/usr/bin/env python3
"""从 Channel1_raw 中提取适合部署到 STM32 的时域特征。

输入文件必须命名为 ``relax_*.csv`` 或 ``bend_*.csv``。
每一个 CSV 文件都被看作一次独立试次（trial），后续划分训练集和验证集时，
可以利用 trial_id 防止同一文件中的重叠窗口被分到不同数据集中。

运行示例：
    python emg_feature_extractor.py
    python emg_feature_extractor.py --target-fs 250 --window-ms 200 \
        --overlap 0.5 --features mav rms wl zc ssc
"""

from __future__ import annotations

# argparse：读取命令行参数，例如 --target-fs 和 --window-ms。
# Path：以跨平台方式处理输入、输出文件路径。
import argparse
from pathlib import Path

# NumPy：数组和特征数值计算。
# pandas：读取原始 CSV、构建特征表并保存 CSV。
# scipy.signal：设计并执行 Butterworth、陷波和 SOS 形式的 IIR 滤波器。
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, tf2sos


# SVM 最终使用数字标签：0 表示放松，1 表示握拳。
LABEL_TO_ID = {"relax": 0, "bend": 1}

# 本脚本允许通过 --features 选择的全部时域特征名称。
AVAILABLE_FEATURES = ("mav", "rms", "wl", "var", "zc", "ssc", "wamp")


def infer_label(path: Path) -> str:
    """根据 CSV 文件名前缀推断标签，例如 bend_001.csv -> bend。"""
    # path.stem 去掉扩展名：
    # bend_001.csv -> bend_001，再按第一个下划线得到 bend。
    label = path.stem.split("_", 1)[0].lower()
    if label not in LABEL_TO_ID:
        raise ValueError(f"Cannot infer relax/bend label from {path.name}")
    return label


def design_filter(
    fs: float,
    low_hz: float,
    high_hz: float,
    notch_hz: float | None,
    order: int,
) -> np.ndarray:
    """设计带通和陷波滤波器，返回级联二阶节（SOS）系数。"""
    # Nyquist frequency = fs/2，所有数字滤波截止频率必须低于它。
    nyquist = fs / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"Require 0 < low < high < Nyquist ({nyquist:g} Hz), got "
            f"{low_hz:g} and {high_hz:g} Hz."
        )

    # Butterworth band-pass：
    # 去除直流偏置/低频漂移，并抑制带宽以外的高频噪声。
    # output="sos" 使用二阶节形式，比直接使用高阶 b/a 系数数值更稳定，？？？？？？ ？
    # 后续也更容易迁移到 STM32 CMSIS-DSP 的 biquad 实现。
    sections = [butter(order, [low_hz, high_hz], btype="bandpass",
                       fs=fs, output="sos")]

    # 如果 notch_hz 不是 None，再串联一个陷波器抑制工频干扰。
    # 国内交流电频率通常为 50 Hz；Q=30 表示陷波带宽较窄。
    if notch_hz is not None:
        if not 0 < notch_hz < nyquist:
            raise ValueError("Notch frequency must be below Nyquist.")
        b, a = iirnotch(notch_hz, Q=30.0, fs=fs)
        sections.append(tf2sos(b, a))

    # 将带通和陷波的多个 SOS 系数纵向拼接，滤波时会依次执行。
    return np.vstack(sections)


def preprocess(
    signal: np.ndarray,
    source_fs: int,
    target_fs: int,
    low_hz: float,
    high_hz: float,
    notch_hz: float | None,
    filter_order: int,
) -> np.ndarray:
    """对一个 CSV 的完整信号执行抗混叠、整数降采样、带通和陷波。"""
    # 当前为了便于 STM32 精确复现，只支持整数倍降采样。
    # 例如 500 -> 250（2倍）、500 -> 125（4倍）可以；
    # 500 -> 300 不能使用当前实现。
    if source_fs % target_fs:
        raise ValueError(
            "For exact STM32 reproduction, target_fs must divide source_fs "
            f"exactly; got {source_fs}/{target_fs}."
        )

    # factor 是降采样倍数，例如 500/250=2。
    factor = source_fs // target_fs

    # 将 ADC 整数值转换成 float64，供 IIR 滤波和特征计算使用。
    x = np.asarray(signal, dtype=np.float64)

    # NaN 或 Inf 会污染后续全部滤波状态和特征，因此提前停止。
    if not np.isfinite(x).all():
        raise ValueError("Signal contains NaN or infinite values.")

    # 降采样之前先做低通抗混叠（anti-aliasing）。
    # 如果直接降采样，超过新 Nyquist frequency 的高频分量会错误折叠到低频。
    # 这种“因果 IIR + 抽取”的实现也较容易用 CMSIS-DSP 在 STM32 上复现。
    if factor > 1:
        # 留出 10% Nyquist 保护带，例如 target_fs=250 时截止为 112.5 Hz。
        anti_alias_cutoff = 0.45 * target_fs
        anti_alias_sos = butter(
            filter_order, anti_alias_cutoff, btype="lowpass",
            fs=source_fs, output="sos"
        )

        # sosfilt 是 causal filter：只使用当前和过去的采样点，
        # 与 STM32 实时滤波一致，不使用未来数据。
        x = sosfilt(anti_alias_sos, x)

        # 每 factor 个点保留一个点。例如 factor=2 时保留 0,2,4,...。
        x = x[::factor]

    # 最终带通上限不能接近或超过降采样后的 Nyquist frequency。
    # 例如 target_fs=125 时，即使 high_hz=100，实际只使用 56.25 Hz。
    effective_high = min(high_hz, 0.45 * target_fs)
    if effective_high <= low_hz:
        raise ValueError(
            f"target_fs={target_fs} is too low for low_hz={low_hz:g}. "
            "Increase target_fs or reduce the band-pass lower cutoff."
        )

    # 在 target_fs 采样率下设计带通/陷波器，并对降采样信号进行因果滤波。
    sos = design_filter(
        target_fs, low_hz, effective_high, notch_hz, filter_order
    )
    return sosfilt(sos, x)


def threshold_crossings(values: np.ndarray, threshold: float) -> int:
    """统计有效过零次数（ZC），忽略幅度小于 threshold 的噪声抖动。"""
    return int(np.count_nonzero(
        # 相邻两点符号不同，说明波形跨过了 0。
        ((values[:-1] >= 0) != (values[1:] >= 0))
        # 同时要求两点的变化足够大，避免把零点附近的小噪声算作过零。
        & (np.abs(values[1:] - values[:-1]) >= threshold)
    ))


def extract_features(
    window: np.ndarray,
    names: list[str],
    threshold: float,
) -> dict[str, float]:
    """为一个信号窗口计算指定的低成本时域特征。"""
    # x 是当前窗口，例如 250 Hz、200 ms 窗口包含 50 个采样点。
    x = np.asarray(window, dtype=np.float64)

    # dx 是一阶差分：dx[i] = x[i+1] - x[i]。
    # WL、SSC 和 WAMP 都会使用相邻采样点的变化量。
    dx = np.diff(x)
    results: dict[str, float] = {}

    for name in names:
        if name == "mav":
            # Mean Absolute Value（平均绝对值）：
            # 反映肌电信号的平均幅度，计算成本很低。
            results[name] = float(np.mean(np.abs(x)))
        elif name == "rms":
            # Root Mean Square（均方根）：
            # 反映信号幅度/能量，对较大幅度采样点比 MAV 更敏感。
            results[name] = float(np.sqrt(np.mean(x * x)))
        elif name == "wl":
            # Waveform Length（波形长度）：
            # 累加相邻采样变化的绝对值，反映幅度和波形复杂程度。
            results[name] = float(np.sum(np.abs(dx)))
        elif name == "var":
            # Variance（样本方差）：
            # 反映信号围绕均值的波动强度；ddof=1 使用 N-1 作分母。
            results[name] = float(np.var(x, ddof=1))
        elif name == "zc":
            # Zero Crossing（过零次数）：
            # 粗略反映频率特性，并用 threshold 抑制噪声过零。
            results[name] = float(threshold_crossings(x, threshold))
        elif name == "ssc":
            # Slope Sign Changes（斜率符号变化次数）：
            # 相邻斜率乘积小于 0，表示发生“上升->下降”或“下降->上升”。
            slopes = dx
            results[name] = float(np.count_nonzero(
                (slopes[:-1] * slopes[1:] < 0)
                & (np.abs(slopes[:-1] - slopes[1:]) >= threshold)
            ))
        elif name == "wamp":
            # Willison Amplitude：
            # 统计相邻采样变化量超过 threshold 的次数。
            results[name] = float(np.count_nonzero(np.abs(dx) >= threshold))
        else:
            raise ValueError(f"Unsupported feature: {name}")
    return results


def process_file(path: Path, args: argparse.Namespace) -> list[dict]:
    """处理一个 CSV：读取 CH1、预处理、滑窗、提取特征并添加标签。"""
    # usecols 保证只读取指定通道，默认只处理 Channel1_raw。
    frame = pd.read_csv(path, usecols=[args.channel])
    raw = frame[args.channel].to_numpy(dtype=np.float64)

    # 每个 CSV 独立执行预处理，因此滤波器状态不会跨越两个不同 trial。
    filtered = preprocess(
        raw,
        source_fs=args.source_fs,
        target_fs=args.target_fs,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        notch_hz=args.notch_hz,
        filter_order=args.filter_order,
    )

    # 把毫秒窗口换算成降采样后的采样点数量：
    # window_samples = window_ms * target_fs / 1000。
    # 例如 200 ms、250 Hz -> 50 samples。
    window_samples = round(args.window_ms * args.target_fs / 1000.0)

    # overlap=0.5 表示 50% 重叠，此时步长为半个窗口。
    step_samples = round(window_samples * (1.0 - args.overlap))

    # 因果 IIR 在每个文件开头从零状态启动，可能产生滤波瞬态。
    # 默认丢弃前 500 ms，不从这段信号生成窗口或分配窗口标签。
    # 因此第一个有效窗口从 0.5 s 开始。
    first_sample = round(args.discard_ms * args.target_fs / 1000.0)
    if window_samples < 2 or step_samples < 1:
        raise ValueError("Window/overlap produces an invalid window step.")

    # 注意：当前是“文件级标签”。bend_001.csv 中的全部窗口都会标为 bend，
    # 即使文件开头实际上还没有开始握拳。这是后续可改进的标签逻辑。
    label = infer_label(path)
    rows: list[dict] = []

    # 从 first_sample 开始，每次移动 step_samples，直到不足一个完整窗口。
    for window_id, start in enumerate(
        range(first_sample, len(filtered) - window_samples + 1, step_samples)
    ):
        stop = start + window_samples

        # 只将当前窗口送入特征提取函数。
        features = extract_features(
            filtered[start:stop], args.features, args.threshold
        )

        # 每个窗口对应输出 CSV 中的一行。
        # source_file/trial_id/window_id/start_s 是元数据，不作为 SVM 特征；
        # 真正喂给 SVM 的是 MAV、RMS、WL、ZC、SSC 等特征列。
        rows.append({
            "source_file": path.name,
            "trial_id": path.stem,
            "window_id": window_id,
            "start_sample": start,
            "end_sample": stop,
            "start_s": start / args.target_fs,
            "label": label,
            "label_id": LABEL_TO_ID[label],
            **features,
        })
    return rows


def parse_args() -> argparse.Namespace:
    """定义并检查所有可从命令行修改的预处理参数。"""
    parser = argparse.ArgumentParser(
        description="Create a window-level Channel1_raw feature dataset."
    )
    # 输入目录中需要直接包含 relax_*.csv 和 bend_*.csv。
    parser.add_argument(
        "--input-dir", type=Path, default=Path("data_saved/data_saved")
    )
    # 所有文件的窗口特征最终合并保存到一个 CSV。
    parser.add_argument(
        "--output", type=Path, default=Path("emg_features.csv")
    )
    # 当前实验默认只读取第一个原始通道。
    parser.add_argument("--channel", default="Channel1_raw")

    # source_fs：仪器原始有效采样率；target_fs：预处理后的目标采样率。
    parser.add_argument("--source-fs", type=int, default=500)
    parser.add_argument("--target-fs", type=int, default=250)

    # window-ms 是单个分类窗口的持续时间。
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument(
        "--discard-ms", type=float, default=500.0,
        help="Discard each trial's initial causal-filter transient."
    )
    parser.add_argument(
        "--overlap", type=float, default=0.5,
        help="Window overlap fraction in [0, 1), e.g. 0.5."
    )

    # 带通滤波参数；默认保留 20-100 Hz，但高频上限还会受 target_fs 限制。
    parser.add_argument("--low-hz", type=float, default=20.0)
    parser.add_argument("--high-hz", type=float, default=100.0)

    # --notch-hz 0 表示关闭陷波，否则默认抑制 50 Hz 工频。
    parser.add_argument(
        "--notch-hz", type=float, default=50.0,
        help="Power-line notch; use 0 to disable."
    )
    parser.add_argument("--filter-order", type=int, default=4)

    # nargs="+" 表示 --features 后可以输入一个或多个特征名称。
    parser.add_argument(
        "--features", nargs="+", choices=AVAILABLE_FEATURES,
        default=["mav", "rms", "wl", "zc", "ssc"]
    )

    # threshold 只作用于 ZC、SSC 和 WAMP，用来抑制小幅噪声。
    parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="Noise threshold in filtered ADC counts for ZC/SSC/WAMP."
    )
    args = parser.parse_args()

    # overlap 必须是 [0,1)：0 表示无重叠，0.5 表示 50% 重叠。
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1).")
    if args.source_fs <= 0 or args.target_fs <= 0:
        parser.error("Sampling rates must be positive.")

    # 将命令行的 0 转成 None，design_filter() 据此跳过陷波器。
    if args.notch_hz == 0:
        args.notch_hz = None
    return args


def main() -> int:
    """程序入口：查找所有数据文件、逐个处理、合并并保存特征表。"""
    args = parse_args()

    # 自动搜索输入目录中的两类 CSV，然后按路径排序，保证运行顺序稳定。
    paths = sorted(
        p for pattern in ("relax_*.csv", "bend_*.csv")
        for p in args.input_dir.glob(pattern)
    )
    if not paths:
        raise FileNotFoundError(
            f"No relax_*.csv or bend_*.csv files in {args.input_dir}"
        )

    # 外层遍历每个文件，内层收集该文件产生的每一个窗口。
    # 每个文件都由 process_file() 独立处理，窗口不会跨越两个 CSV。
    rows = [row for path in paths for row in process_file(path, args)]
    result = pd.DataFrame(rows)

    # 如果输出目录还不存在，先递归创建；index=False 避免额外保存行号列。
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    # 在终端汇报处理规模、两类窗口数量、特征组合和输出位置。
    counts = result.groupby("label").size().to_dict()
    print(f"Processed {len(paths)} trials -> {len(result)} windows")
    print(f"Class counts: {counts}")
    print(f"Features: {args.features}")
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    # 只有直接运行本文件时才执行 main()；被其他文件 import 时不会自动运行。
    raise SystemExit(main())
