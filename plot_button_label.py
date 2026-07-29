#!/usr/bin/env python3
"""按接收时间戳绘制 Channel1_raw 波形和 button_label。

用法示例：
    python plot_button_label.py session_1.csv
    python plot_button_label.py data_with_button/session_2.csv
    python plot_button_label.py session_3.csv --save session_3_label_check.png

如果不传文件名，脚本会在终端中提示输入。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as error:
    raise SystemExit(
        f"缺少 Python 依赖 {error.name!r}。请先执行：\n"
        "  python -m pip install matplotlib numpy pandas"
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data_with_button"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot button_label and Channel1_raw against host_receive_time "
            "to inspect label alignment."
        )
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help=(
            "CSV 文件名或路径，例如 session_1.csv。"
            "只写文件名时默认从 data_with_button 中查找。"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"默认数据目录（默认：{DEFAULT_DATA_DIR}）。",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="将图片保存到指定路径，例如 session_1_label_check.png。",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出绘图窗口；通常与 --save 一起使用。",
    )
    return parser.parse_args()


def choose_filename(data_dir: Path) -> str:
    """未提供命令行参数时，显示可用文件并提示用户输入。"""
    available = sorted(data_dir.glob("session_*.csv"))
    if available:
        print("可用 CSV：")
        for path in available:
            print(f"  - {path.name}")
    return input("请输入要绘制的 CSV 文件名：").strip()


def resolve_csv_path(filename: str, data_dir: Path) -> Path:
    """支持完整路径、相对路径，以及 data_with_button 下的文件名。"""
    if not filename:
        raise ValueError("文件名不能为空。")

    candidate = Path(filename).expanduser()
    candidates = [candidate]
    if candidate.suffix.lower() != ".csv":
        candidates.append(candidate.with_suffix(".csv"))

    if not candidate.is_absolute():
        relative_candidates = list(candidates)
        candidates.extend(SCRIPT_DIR / path for path in relative_candidates)
        candidates.extend(data_dir / path.name for path in relative_candidates)

    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path.resolve()

    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"找不到 CSV 文件，已尝试：\n{attempted}")


def normalized_column_name(name: str) -> str:
    """忽略大小写、空格和下划线比较列名。"""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_column(columns: list[str], requested: str) -> str:
    """找到实际列名，例如 channel_1raw 可匹配 Channel1_raw。"""
    target = normalized_column_name(requested)
    matches = [name for name in columns if normalized_column_name(name) == target]
    if not matches:
        raise KeyError(
            f"CSV 中缺少 {requested!r} 列。实际列名为：{', '.join(columns)}"
        )
    return matches[0]


def load_plot_data(csv_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """只读取绘图需要的三列，并清理无效行。"""
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    timestamp_column = find_column(columns, "host_receive_time")
    label_column = find_column(columns, "button_label")
    raw_column = find_column(columns, "channel_1raw")

    data = pd.read_csv(
        csv_path,
        usecols=[timestamp_column, label_column, raw_column],
    )
    data[timestamp_column] = pd.to_datetime(
        data[timestamp_column],
        errors="coerce",
    )
    data[label_column] = pd.to_numeric(data[label_column], errors="coerce")
    data[raw_column] = pd.to_numeric(data[raw_column], errors="coerce")

    invalid_rows = data[
        [timestamp_column, label_column, raw_column]
    ].isna().any(axis=1)
    if invalid_rows.any():
        print(f"提示：跳过 {int(invalid_rows.sum())} 行无效数据。")
        data = data.loc[~invalid_rows].copy()

    if data.empty:
        raise ValueError("CSV 中没有可绘制的有效数据。")

    data.sort_values(timestamp_column, kind="stable", inplace=True)
    return data, timestamp_column, label_column, raw_column


def plot_data(
    data: pd.DataFrame,
    csv_path: Path,
    timestamp_column: str,
    label_column: str,
    raw_column: str,
) -> plt.Figure:
    timestamps = data[timestamp_column]
    raw = data[raw_column].to_numpy()
    labels = data[label_column].to_numpy()

    duration_s = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
    figure, (wave_axis, label_axis) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    wave_axis.plot(
        timestamps,
        raw,
        color="#1769aa",
        linewidth=0.65,
        label=raw_column,
    )
    # 在原始波形上标出按键处于激活状态的时间段，方便检查对齐。
    wave_axis.fill_between(
        timestamps,
        0,
        1,
        where=labels > 0,
        step="post",
        color="#ffb300",
        alpha=0.18,
        transform=wave_axis.get_xaxis_transform(),
        label=f"{label_column} > 0",
    )
    wave_axis.set_ylabel(raw_column)
    wave_axis.set_title(
        f"{csv_path.name} | {len(data):,} samples | {duration_s:.3f} s"
    )
    wave_axis.grid(True, alpha=0.25)
    wave_axis.legend(loc="upper right")

    label_axis.step(
        timestamps,
        labels,
        where="post",
        color="#d84315",
        linewidth=1.2,
    )
    label_axis.fill_between(
        timestamps,
        0,
        labels,
        step="post",
        color="#ffb300",
        alpha=0.28,
    )
    label_axis.set_ylabel(label_column)
    label_axis.set_xlabel(timestamp_column)
    label_axis.grid(True, alpha=0.25)

    unique_labels = np.unique(labels)
    if len(unique_labels) <= 12:
        label_axis.set_yticks(unique_labels)
    if np.all(unique_labels >= 0):
        label_axis.set_ylim(
            bottom=min(-0.1, float(unique_labels.min()) - 0.1),
            top=max(1.1, float(unique_labels.max()) + 0.1),
        )

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    label_axis.xaxis.set_major_locator(locator)
    label_axis.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )

    figure.autofmt_xdate(rotation=20)
    figure.tight_layout()
    return figure


def main() -> int:
    args = parse_args()
    try:
        filename = args.filename or choose_filename(args.data_dir)
        csv_path = resolve_csv_path(filename, args.data_dir)
        data, timestamp_column, label_column, raw_column = load_plot_data(csv_path)
        figure = plot_data(
            data,
            csv_path,
            timestamp_column,
            label_column,
            raw_column,
        )

        if args.save:
            output_path = args.save.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=180, bbox_inches="tight")
            print(f"图片已保存：{output_path}")

        print(
            f"已绘制：{csv_path.name}（{len(data):,} 个采样点，"
            f"标签值 {sorted(data[label_column].unique().tolist())}）"
        )
        if not args.no_show:
            plt.show()
        else:
            plt.close(figure)
        return 0
    except (FileNotFoundError, KeyError, ValueError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
