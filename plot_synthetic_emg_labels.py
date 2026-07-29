#!/usr/bin/env python3
"""绘制模拟 sEMG 原始波形及其按钮标签。

默认不需要参数：脚本会自动寻找 ``quick_validation_artifacts`` 中最近一次
生成的第一个训练 CSV，并把 PNG 保存到该 CSV 旁边。

示例：

    python plot_synthetic_emg_labels.py

    python plot_synthetic_emg_labels.py \
        --input path/to/synthetic_train_session_001.csv \
        --start-s 2 --duration-s 5

    python plot_synthetic_emg_labels.py --input-dir fake_data

图中：

* 上图是 ``Channel1_raw``；
* 下图蓝线是用于训练的 ``button_label``；
* 如果CSV包含 ``sim_true_label``，橙色虚线表示模拟隐藏真值；
* 红色区域表示按钮标签与隐藏真值不一致；
* 灰色区域表示按钮上升/下降沿前后各 ``--transition-band-ms`` 的审计范围；
  这些窗口仍会保留。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SAMPLING_RATE = 500.0


def find_latest_synthetic_csv(output_root: Path) -> Path:
    """寻找最近一次quick validation生成的第一个训练Session。"""
    if not output_root.is_dir():
        raise FileNotFoundError(
            f"Synthetic output directory not found: {output_root}"
        )

    run_directories = sorted(
        (
            path
            for path in output_root.glob("run_*")
            if path.is_dir()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_directory in run_directories:
        paths = sorted(
            (run_directory / "data" / "train").glob("*.csv")
        )
        if paths:
            return paths[0]

    raise FileNotFoundError(
        f"No synthetic training CSV found below: {output_root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot synthetic Channel1_raw and button labels."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        type=Path,
        help=(
            "Synthetic Session CSV. Default: first training CSV from the "
            "latest quick_validation_artifacts run."
        ),
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help=(
            "Recursively plot every CSV below this directory. Each PNG is "
            "saved beside its source CSV."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("quick_validation_artifacts"),
        help="Used only when --input is omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path. Default: <input_stem>_waveform_labels.png.",
    )
    parser.add_argument("--sampling-rate", type=float, default=500.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        help="Time span to plot. Default: from start-s to end of CSV.",
    )
    parser.add_argument(
        "--transition-band-ms",
        "--guard-ms",
        dest="transition_band_ms",
        type=float,
        default=50.0,
        help="Shade this interval on both sides of every button edge.",
    )
    parser.add_argument(
        "--hide-true-label",
        action="store_true",
        help="Do not plot sim_true_label even when the column exists.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive plot window after saving.",
    )
    parser.add_argument("--width-px", type=int, default=1800)
    args = parser.parse_args()

    if args.sampling_rate <= 0.0:
        parser.error("--sampling-rate must be positive.")
    if args.start_s < 0.0:
        parser.error("--start-s cannot be negative.")
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive.")
    if args.transition_band_ms < 0.0:
        parser.error("--transition-band-ms cannot be negative.")
    if args.width_px < 800:
        parser.error("--width-px must be at least 800.")
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            parser.error(f"--input-dir not found: {args.input_dir}")
        if args.output is not None:
            parser.error("--output cannot be used together with --input-dir.")
        if args.show:
            parser.error("--show cannot be used together with --input-dir.")
    return args


def plot_directory(args: argparse.Namespace) -> int:
    """递归调用单文件模式，为目录中的每个CSV生成一张同名图。"""
    input_paths = sorted(args.input_dir.rglob("*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No CSV files found below: {args.input_dir}")

    print(
        f"Batch plotting {len(input_paths)} CSV files below "
        f"{args.input_dir.resolve()}..."
    )
    for index, input_path in enumerate(input_paths, start=1):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input",
            str(input_path.resolve()),
            "--sampling-rate",
            str(args.sampling_rate),
            "--start-s",
            str(args.start_s),
            "--transition-band-ms",
            str(args.transition_band_ms),
            "--width-px",
            str(args.width_px),
        ]
        if args.duration_s is not None:
            command.extend(["--duration-s", str(args.duration_s)])
        if args.hide_true_label:
            command.append("--hide-true-label")
        print(f"[{index}/{len(input_paths)}] {input_path}")
        subprocess.run(command, check=True)

    print(f"Batch complete: generated {len(input_paths)} PNG files.")
    return 0


def read_plot_range(
    path: Path,
    sampling_rate: float,
    start_s: float,
    duration_s: float | None,
) -> tuple[pd.DataFrame, np.ndarray, float, float]:
    """读取CSV并截取用户指定的时间范围。"""
    frame = pd.read_csv(path)
    required = {"Channel1_raw", "button_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{path}: CSV is empty.")

    if "recording_elapsed_s" in frame.columns:
        time_s = pd.to_numeric(
            frame["recording_elapsed_s"], errors="raise"
        ).to_numpy(dtype=np.float64)
    elif "elapsed_s" in frame.columns:
        elapsed = pd.to_numeric(
            frame["elapsed_s"], errors="raise"
        ).to_numpy(dtype=np.float64)
        time_s = elapsed - elapsed[0]
    else:
        time_s = np.arange(len(frame), dtype=np.float64) / sampling_rate

    if not np.isfinite(time_s).all():
        raise ValueError(f"{path}: time column contains NaN/Inf.")
    end_s = (
        float(time_s[-1]) + 1.0 / sampling_rate
        if duration_s is None
        else start_s + duration_s
    )
    mask = (time_s >= start_s) & (time_s < end_s)
    if not np.any(mask):
        raise ValueError(
            f"No samples lie in requested range {start_s:g}--{end_s:g} s."
        )

    selected = frame.loc[mask].copy()
    selected_time = time_s[mask]
    return selected, selected_time, float(selected_time[0]), float(
        selected_time[-1]
    )


def button_edges(time_s: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """返回按钮发生0↔1变化时对应的时间。"""
    edge_indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return time_s[edge_indices]


def step_coordinates(
    time_s: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把逐采样标签转换成不依赖绘图库stepMode的阶梯线坐标。"""
    if time_s.size != values.size:
        raise ValueError("Time and label arrays must have the same length.")
    if time_s.size == 1:
        return time_s.copy(), values.astype(np.float64)
    step_time = np.repeat(time_s, 2)[1:]
    step_values = np.repeat(values.astype(np.float64), 2)[:-1]
    return step_time, step_values


def true_regions(
    mask: np.ndarray,
    time_s: np.ndarray,
    sampling_rate: float,
) -> list[tuple[float, float]]:
    """把连续True mask转换成若干时间区间，供红色错位区域绘制。"""
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    regions: list[tuple[float, float]] = []
    for start_index, stop_index in changes.reshape(-1, 2):
        start_s = float(time_s[start_index])
        stop_s = (
            float(time_s[stop_index - 1]) + 1.0 / sampling_rate
        )
        regions.append((start_s, stop_s))
    return regions


def load_export_font(application: object, qt_gui: object) -> str:
    """为Qt offscreen导出显式加载字体，避免PNG文字显示成方框。"""
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if not path.is_file():
            continue
        font_id = qt_gui.QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = qt_gui.QFontDatabase.applicationFontFamilies(font_id)
        if families:
            family = str(families[0])
            application.setFont(qt_gui.QFont(family, 9))
            return family
    raise RuntimeError(
        "No usable font was found for PNG export. Install Arial or "
        "DejaVu Sans, or edit load_export_font() with a local .ttf path."
    )


def main() -> int:
    args = parse_args()
    if args.input_dir is not None:
        return plot_directory(args)

    input_path = args.input or find_latest_synthetic_csv(args.output_root)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_waveform_labels.png"
    )
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must use the .png extension.")

    # 项目已有PySide6 + PyQtGraph依赖，因此不额外要求安装Matplotlib。
    # 非交互模式使用Qt offscreen平台，在PowerShell中也能直接导出PNG。
    if not args.show:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import pyqtgraph as pg
    import pyqtgraph.exporters
    from PySide6 import QtCore, QtGui, QtWidgets

    frame, time_s, shown_start_s, shown_end_s = read_plot_range(
        input_path,
        sampling_rate=args.sampling_rate,
        start_s=args.start_s,
        duration_s=args.duration_s,
    )
    raw = pd.to_numeric(
        frame["Channel1_raw"], errors="raise"
    ).to_numpy(dtype=np.float64)
    label = pd.to_numeric(
        frame["button_label"], errors="raise"
    ).to_numpy(dtype=np.uint8)
    if not np.isfinite(raw).all():
        raise ValueError("Channel1_raw contains NaN/Inf.")
    if not set(np.unique(label)).issubset({0, 1}):
        raise ValueError("button_label must contain only 0/1.")

    show_true_label = (
        not args.hide_true_label and "sim_true_label" in frame.columns
    )
    true_label: np.ndarray | None = None
    if show_true_label:
        true_label = pd.to_numeric(
            frame["sim_true_label"], errors="raise"
        ).to_numpy(dtype=np.uint8)
        if not set(np.unique(true_label)).issubset({0, 1}):
            raise ValueError("sim_true_label must contain only 0/1.")

    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])
    font_family = load_export_font(application, QtGui)
    pg.setConfigOptions(
        background="w",
        foreground="k",
        antialias=True,
    )
    widget = pg.GraphicsLayoutWidget(
        title="Synthetic sEMG waveform and button labels"
    )
    widget.resize(1400, 720)
    widget.addLabel(
        (
            f"<div style='font-family:{font_family}'>"
            "<b>Synthetic sEMG waveform and labels</b><br>"
            f"<span style='font-size:10pt'>{input_path.name}</span></div>"
        ),
        row=0,
        col=0,
    )
    wave_plot = widget.addPlot(row=1, col=0)
    label_plot = widget.addPlot(row=2, col=0)
    label_plot.setXLink(wave_plot)

    wave_plot.plot(
        time_s,
        raw,
        pen=pg.mkPen("#1756A9", width=1.0),
        name="Channel1_raw",
    )
    baseline = float(np.median(raw))
    baseline_line = pg.InfiniteLine(
        pos=baseline,
        angle=0,
        pen=pg.mkPen("#555555", width=1, style=QtCore.Qt.DotLine),
        label=f"median={baseline:.1f}",
        labelOpts={"position": 0.95, "color": "#555555"},
    )
    wave_plot.addItem(baseline_line)
    # 关闭 SI 前缀自动缩放，纵轴直接显示原始 ADC 数值（例如 1000，而不是 1.0 k）。
    wave_plot.getAxis("left").enableAutoSIPrefix(False)
    wave_plot.setLabel("left", "Channel1_raw (ADC counts)")
    wave_plot.showGrid(x=True, y=True, alpha=0.18)
    wave_plot.addLegend(offset=(-10, 10))

    button_step_time, button_step_value = step_coordinates(time_s, label)
    label_plot.plot(
        button_step_time,
        button_step_value,
        pen=pg.mkPen("#1769AA", width=2.2),
        name="button_label (training label)",
    )
    if true_label is not None:
        true_step_time, true_step_value = step_coordinates(
            time_s, true_label
        )
        label_plot.plot(
            true_step_time,
            true_step_value,
            pen=pg.mkPen(
                "#E67E22",
                width=2.0,
                style=QtCore.Qt.DashLine,
            ),
            name="sim_true_label (hidden truth)",
        )
        mismatch = label != true_label
        for start_s, stop_s in true_regions(
            mismatch, time_s, args.sampling_rate
        ):
            mismatch_region = pg.LinearRegionItem(
                values=(start_s, stop_s),
                orientation="vertical",
                movable=False,
                brush=pg.mkBrush(231, 76, 60, 65),
                pen=pg.mkPen(None),
            )
            label_plot.addItem(mismatch_region)

    edge_times = button_edges(time_s, label)
    transition_band_s = args.transition_band_ms / 1000.0
    for edge_time in edge_times:
        transition_band_region = pg.LinearRegionItem(
            values=(
                max(shown_start_s, float(edge_time) - transition_band_s),
                min(shown_end_s, float(edge_time) + transition_band_s),
            ),
            orientation="vertical",
            movable=False,
            brush=pg.mkBrush(127, 140, 141, 42),
            pen=pg.mkPen(None),
        )
        label_plot.addItem(transition_band_region)
        wave_plot.addItem(
            pg.InfiniteLine(
                pos=float(edge_time),
                angle=90,
                pen=pg.mkPen(127, 140, 141, 75, width=1),
            )
        )

    label_plot.setYRange(-0.12, 1.12, padding=0)
    label_plot.getAxis("left").setTicks(
        [[(0.0, "Relax (0)"), (1.0, "Clench (1)")]]
    )
    label_plot.getAxis("left").setWidth(78)
    label_plot.setLabel("left", "State")
    label_plot.setLabel("bottom", "Time", units="s")
    label_plot.showGrid(x=True, y=False, alpha=0.18)
    label_plot.addLegend(offset=(-10, 10))
    label_plot.setTitle(
        (
            f"<div style='font-family:{font_family}'>"
            "<span style='color:#1769AA'>blue: button label</span>; "
            "<span style='color:#E67E22'>orange: hidden truth</span>; "
            "<span style='color:#E74C3C'>red: mismatch</span>; "
            "gray: +/-"
            f"{args.transition_band_ms:g} ms transition band</div>"
        )
    )
    wave_plot.setXRange(shown_start_s, shown_end_s, padding=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # show()也会在offscreen模式中完成布局，但不会弹出可见窗口。
    widget.show()
    application.processEvents()
    exporter = pyqtgraph.exporters.ImageExporter(widget.scene())
    exporter.parameters()["width"] = args.width_px
    exporter.export(str(output_path.resolve()))
    print(f"Input CSV: {input_path.resolve()}")
    print(
        f"Plotted range: {shown_start_s:.3f}--{shown_end_s:.3f} s "
        f"({len(frame)} samples)"
    )
    print(f"Button edges in range: {len(edge_times)}")
    if true_label is not None:
        mismatch_samples = int(np.count_nonzero(label != true_label))
        print(
            "Button/true mismatch in plotted range: "
            f"{mismatch_samples} samples "
            f"({mismatch_samples / len(frame):.2%})"
        )
    print(f"Saved PNG: {output_path.resolve()}")

    if args.show:
        application.exec()
    else:
        widget.close()
    if owns_application:
        application.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
