#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG 数据采集工具
================
- 蓝牙串口读取淘宝 EMG 采集板数据 (AA BB + 4×int16 LE)
- 仅保留 Channel1
- 每 64 点 = 1 bin (128ms), 按钮实时标注 fist/relax
- 边采边存: session_XXX_raw.bin + session_XXX_labels.csv

用法:
    python collect_gui.py
    python collect_gui.py --port COM3
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
from serial.tools import list_ports

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

FRAME_HEADER     = b"\xAA\xBB"
FRAME_SIZE       = 10          # 2 header + 8 payload
PAYLOAD_SIZE     = 8
NUM_CHANNELS     = 4
CHANNEL_INDEX    = 0           # 只用 Channel1
BIN_SIZE         = 64          # 64 点/bin
SAMPLE_RATE      = 500         # Hz
BAUD_RATE        = 115200

LABEL_FIST       = 0
LABEL_RELAX      = 1

# ═══════════════════════════════════════════════════════════
#  帧解析器
# ═══════════════════════════════════════════════════════════

class FrameParser:
    """流式解析 AA BB + 4×int16 LE 帧 (无校验)"""

    def __init__(self):
        self.buffer = bytearray()
        self.sample_struct = struct.Struct("<hhhh")  # 4 × int16 LE
        self.discarded = 0
        self.parsed = 0

    def feed(self, data: bytes) -> np.ndarray:
        """喂入原始字节, 返回解析出的 int16 数组 [N×4]"""
        if not data:
            return np.empty((0, NUM_CHANNELS), dtype=np.int16)

        self.buffer.extend(data)
        decoded: list[tuple] = []

        while True:
            idx = self.buffer.find(FRAME_HEADER)
            if idx < 0:
                keep = max(0, len(FRAME_HEADER) - 1)
                if keep and len(self.buffer) >= keep:
                    self.discarded += len(self.buffer) - keep
                    self.buffer[:] = self.buffer[-keep:]
                else:
                    self.discarded += len(self.buffer)
                    self.buffer.clear()
                break

            if idx > 0:
                self.discarded += idx
                del self.buffer[:idx]

            if len(self.buffer) < FRAME_SIZE:
                break

            payload = bytes(self.buffer[2:10])
            try:
                values = self.sample_struct.unpack(payload)
                decoded.append(values)
                self.parsed += 1
            except struct.error:
                self.discarded += 1
                del self.buffer[0]
                continue

            del self.buffer[:FRAME_SIZE]

        return np.array(decoded, dtype=np.int16) if decoded else np.empty((0, NUM_CHANNELS), dtype=np.int16)


# ═══════════════════════════════════════════════════════════
#  串口读取线程
# ═══════════════════════════════════════════════════════════

class SerialReader(QtCore.QThread):
    samples_ready = QtCore.Signal(object)       # np.ndarray [N×4] int16
    connected     = QtCore.Signal(str)
    disconnected  = QtCore.Signal()
    error_occurred = QtCore.Signal(str)

    def __init__(self, port: str):
        super().__init__()
        self.port = port
        self._running = False
        self._serial: serial.Serial | None = None
        self.parser = FrameParser()

    def run(self):
        self._running = True
        self.parser = FrameParser()
        try:
            self._serial = serial.Serial(
                port=self.port, baudrate=BAUD_RATE,
                timeout=0.05, write_timeout=0.5)
            self._serial.reset_input_buffer()
            self.connected.emit(self.port)

            while self._running:
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting > 0 else 1)
                if chunk:
                    frames = self.parser.feed(chunk)
                    if frames.shape[0]:
                        self.samples_ready.emit(frames)
        except (serial.SerialException, OSError) as e:
            if self._running:
                self.error_occurred.emit(str(e))
        finally:
            self._running = False
            if self._serial and self._serial.is_open:
                try:
                    self._serial.close()
                except serial.SerialException:
                    pass
            self.disconnected.emit()

    def stop(self):
        self._running = False
        if self._serial:
            try:
                self._serial.cancel_read()
            except (AttributeError, serial.SerialException):
                pass


# ═══════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, requested_port: str | None = None):
        super().__init__()
        self.setWindowTitle("EMG 数据采集 — collect_gui")
        self.resize(1100, 720)

        # ── 状态 ──
        self.reader: SerialReader | None = None
        self._connected = False
        self._recording = False

        # 按钮状态: 0=fist, 1=relax (初始 relax)
        self._label_state = LABEL_RELAX

        # bin 计数
        self._fist_bins = 0
        self._relax_bins = 0

        # 当前 bin 缓冲区
        self._bin_buf: list[int] = []          # int16 值
        self._bin_start_sample = 0             # 这个 bin 起始采样点索引
        self._total_samples = 0                # 总采样点计数

        # 波形显示 buffer (最近 5000 点)
        self._plot_data = np.zeros(5000, dtype=float)
        self._plot_write = 0
        self._plot_count = 0

        # 文件句柄
        self._raw_file = None
        self._csv_file = None
        self._csv_writer = None
        self._session_name = ""

        # ── UI ──
        self._build_ui()
        self.refresh_ports()
        if requested_port:
            self.port_box.setCurrentText(requested_port)

        # 波形刷新定时器 (30fps)
        self._plot_timer = QtCore.QTimer(self)
        self._plot_timer.setInterval(33)
        self._plot_timer.timeout.connect(self._update_plot)
        self._plot_timer.start()

    # ── UI 构建 ──────────────────────────────────

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # ──── 第1行: 串口 ────
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("串口:"))
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setEditable(True)
        self.port_box.setMinimumWidth(120)
        row1.addWidget(self.port_box)

        self.refresh_btn = QtWidgets.QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh_ports)
        row1.addWidget(self.refresh_btn)

        row1.addWidget(QtWidgets.QLabel("波特率: 115200"))
        row1.addStretch()

        self.connect_btn = QtWidgets.QPushButton("连接")
        self.connect_btn.setMinimumWidth(100)
        self.connect_btn.clicked.connect(self.toggle_connection)
        row1.addWidget(self.connect_btn)

        self.status_label = QtWidgets.QLabel("未连接")
        row1.addWidget(self.status_label)
        layout.addLayout(row1)

        # ──── 第2行: 动作切换按钮 + 指示 ────
        row2 = QtWidgets.QHBoxLayout()

        self.action_btn = QtWidgets.QPushButton("🖐 放松 → 点击切换握拳")
        self.action_btn.setMinimumHeight(60)
        action_font = self.action_btn.font()
        action_font.setPointSize(18)
        action_font.setBold(True)
        self.action_btn.setFont(action_font)
        self.action_btn.clicked.connect(self._toggle_label)
        self.action_btn.setEnabled(False)
        row2.addWidget(self.action_btn)

        layout.addLayout(row2)

        # ──── 第3行: bin 计数 ────
        row3 = QtWidgets.QHBoxLayout()
        row3.addStretch()

        self.fist_count_label = QtWidgets.QLabel("Fist: 0")
        self.fist_count_label.setMinimumWidth(100)
        ffont = self.fist_count_label.font()
        ffont.setPointSize(14)
        ffont.setBold(True)
        self.fist_count_label.setFont(ffont)
        self.fist_count_label.setStyleSheet("color: #ff4444;")
        row3.addWidget(self.fist_count_label)

        row3.addWidget(QtWidgets.QLabel("  |  "))

        self.relax_count_label = QtWidgets.QLabel("Relax: 0")
        self.relax_count_label.setMinimumWidth(120)
        self.relax_count_label.setFont(ffont)
        self.relax_count_label.setStyleSheet("color: #4444ff;")
        row3.addWidget(self.relax_count_label)

        row3.addWidget(QtWidgets.QLabel("  |  "))

        self.total_label = QtWidgets.QLabel("总计: 0 bins")
        self.total_label.setFont(ffont)
        row3.addWidget(self.total_label)

        # 达标指示
        self.fist_check = QtWidgets.QLabel("○")
        self.fist_check.setFont(ffont)
        row3.addWidget(self.fist_check)
        self.relax_check = QtWidgets.QLabel("○")
        self.relax_check.setFont(ffont)
        row3.addWidget(self.relax_check)

        row3.addStretch()
        layout.addLayout(row3)

        # ──── Y轴控制 (与 emg_usart_viewer 一致) ────
        y_row = QtWidgets.QHBoxLayout()
        y_row.addWidget(QtWidgets.QLabel("Y 轴:"))

        self.auto_y_check = QtWidgets.QCheckBox("自动范围")
        self.auto_y_check.setChecked(False)
        self.auto_y_check.stateChanged.connect(self._on_auto_y_changed)
        y_row.addWidget(self.auto_y_check)

        y_row.addWidget(QtWidgets.QLabel("Min:"))
        self.y_min_box = QtWidgets.QDoubleSpinBox()
        self.y_min_box.setRange(-32768, 32767)
        self.y_min_box.setDecimals(0)
        self.y_min_box.setValue(-500)
        self.y_min_box.valueChanged.connect(self._on_manual_y_changed)
        y_row.addWidget(self.y_min_box)

        y_row.addWidget(QtWidgets.QLabel("Max:"))
        self.y_max_box = QtWidgets.QDoubleSpinBox()
        self.y_max_box.setRange(-32768, 32767)
        self.y_max_box.setDecimals(0)
        self.y_max_box.setValue(2000)
        self.y_max_box.valueChanged.connect(self._on_manual_y_changed)
        y_row.addWidget(self.y_max_box)

        reset_btn = QtWidgets.QPushButton("重置视图")
        reset_btn.clicked.connect(self._reset_view)
        y_row.addWidget(reset_btn)
        y_row.addStretch()
        layout.addLayout(y_row)

        # ──── 波形 ────
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("k")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel("bottom", "采样点 (最近5000点)")
        self.plot_widget.setLabel("left", "ADC 值")
        self.plot_widget.setYRange(-500, 2000)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setDownsampling(auto=True, mode="peak")
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color="#00ff88", width=1.2))
        layout.addWidget(self.plot_widget, stretch=1)

        # ──── 底部: 记录控制 ────
        row4 = QtWidgets.QHBoxLayout()
        self.record_btn = QtWidgets.QPushButton("⏺ 开始记录")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self._start_recording)
        row4.addWidget(self.record_btn)

        self.stop_btn = QtWidgets.QPushButton("⏹ 停止并保存")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_recording)
        row4.addWidget(self.stop_btn)

        self.rec_status = QtWidgets.QLabel("")
        row4.addWidget(self.rec_status)
        row4.addStretch()

        self.replay_btn = QtWidgets.QPushButton("📂 回放已保存的数据")
        self.replay_btn.clicked.connect(self._replay_saved)
        row4.addWidget(self.replay_btn)

        self.sample_label = QtWidgets.QLabel("采样点: 0")
        row4.addWidget(self.sample_label)
        layout.addLayout(row4)

    # ── 串口 ────────────────────────────────────

    @QtCore.Slot()
    def refresh_ports(self):
        current = self.port_box.currentText().strip()
        ports = sorted(p.device for p in list_ports.comports())
        self.port_box.blockSignals(True)
        self.port_box.clear()
        self.port_box.addItems(ports)
        if ports:
            self.port_box.setCurrentText(current or ports[0])
        self.port_box.blockSignals(False)

    @QtCore.Slot()
    def toggle_connection(self):
        if self._connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_box.currentText().strip()
        if not port:
            return
        self.connect_btn.setEnabled(False)
        self.status_label.setText(f"连接中 {port}...")
        self.reader = SerialReader(port)
        self.reader.samples_ready.connect(self._on_samples)
        self.reader.connected.connect(self._on_connected)
        self.reader.disconnected.connect(self._on_disconnected)
        self.reader.error_occurred.connect(self._on_error)
        self.reader.start()

    def _disconnect(self):
        if self.reader:
            self.reader.stop()
            self.reader.wait(1500)

    @QtCore.Slot(str)
    def _on_connected(self, port: str):
        self._connected = True
        self.connect_btn.setText("断开")
        self.connect_btn.setEnabled(True)
        self.port_box.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.action_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.status_label.setText(f"已连接: {port}")

    @QtCore.Slot()
    def _on_disconnected(self):
        self._connected = False
        self._recording = False
        self.connect_btn.setText("连接")
        self.connect_btn.setEnabled(True)
        self.port_box.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.action_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已断开")

    @QtCore.Slot(str)
    def _on_error(self, msg: str):
        self.status_label.setText(f"错误: {msg}")

    # ── 数据接收 ────────────────────────────────

    @QtCore.Slot(object)
    def _on_samples(self, frames: np.ndarray):
        """收到解析好的帧 [N×4] int16"""
        if frames.size == 0:
            return

        ch1_raw = frames[:, CHANNEL_INDEX].astype(np.int16)

        for raw_val in ch1_raw:
            raw = int(raw_val)
            display = float(raw)   # 显示原始 ADC 值 (~1000)

            # 追加 raw.bin (存原始 int16)
            if self._raw_file:
                self._raw_file.write(struct.pack("<h", raw))

            # 当前 bin
            self._bin_buf.append(raw)

            # 波形 (显示减去偏移后的值)
            self._plot_data[self._plot_write] = display
            self._plot_write = (self._plot_write + 1) % 5000
            self._plot_count += 1

            # 采样计数
            self._total_samples += 1

            # bin 满了?
            if len(self._bin_buf) >= BIN_SIZE:
                self._finish_bin()

        self.sample_label.setText(f"采样点: {self._total_samples}")

    def _finish_bin(self):
        """当前 bin 攒满 64 点, 写 labels.csv"""
        end_sample = self._total_samples - 1
        start_sample = end_sample - BIN_SIZE + 1
        bin_index = (start_sample) // BIN_SIZE

        # 写 labels.csv
        if self._csv_writer:
            self._csv_writer.writerow([bin_index, start_sample, end_sample, self._label_state])
            # 定期 flush
            if bin_index % 10 == 0 and self._csv_file:
                self._csv_file.flush()

        # 计数
        if self._label_state == LABEL_FIST:
            self._fist_bins += 1
        else:
            self._relax_bins += 1

        # 清空当前 bin
        self._bin_buf.clear()
        self._bin_start_sample = self._total_samples

        # 更新 UI
        self._update_counts()

    def _update_counts(self):
        self.fist_count_label.setText(f"Fist: {self._fist_bins}")
        self.relax_count_label.setText(f"Relax: {self._relax_bins}")
        self.total_label.setText(f"总计: {self._fist_bins + self._relax_bins} bins")

        # 达标指示 (≥100 bin)
        self.fist_check.setText("✅" if self._fist_bins >= 100 else "○")
        self.relax_check.setText("✅" if self._relax_bins >= 100 else "○")

    # ── 按钮 ────────────────────────────────────

    @QtCore.Slot()
    def _toggle_label(self):
        """切换 fist ↔ relax"""
        if self._label_state == LABEL_RELAX:
            self._label_state = LABEL_FIST
            self.action_btn.setText("💪 握拳 → 点击切换放松")
            self.action_btn.setStyleSheet("color: #ff4444;")
            self.rec_status.setText("当前动作: 握拳")
        else:
            self._label_state = LABEL_RELAX
            self.action_btn.setText("🖐 放松 → 点击切换握拳")
            self.action_btn.setStyleSheet("color: #4444ff;")
            self.rec_status.setText("当前动作: 放松")

    @QtCore.Slot()
    def _start_recording(self):
        """开始记录: 打开 raw.bin + labels.csv"""
        # 自动编号
        script_dir = Path(__file__).resolve().parent
        save_dir = script_dir / "data_recorded"
        save_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(save_dir.glob("session_*_raw.bin"))
        next_num = len(existing) + 1
        self._session_name = f"session_{next_num:03d}"

        raw_path = save_dir / f"{self._session_name}_raw.bin"
        csv_path = save_dir / f"{self._session_name}_labels.csv"

        try:
            self._raw_file = open(str(raw_path), "wb")
            self._csv_file = open(str(csv_path), "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["bin_index", "start_row", "end_row", "label"])
            self._csv_file.flush()
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "文件错误", str(e))
            self._raw_file = None
            self._csv_file = None
            return

        # 重置
        self._fist_bins = 0
        self._relax_bins = 0
        self._bin_buf.clear()
        self._bin_start_sample = 0
        self._total_samples = 0
        self._plot_write = 0
        self._plot_count = 0
        self._plot_data.fill(0)

        self._recording = True
        self.record_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.rec_status.setText(f"⏺ 记录中 → {self._session_name}")
        self.status_label.setText(f"记录: {self._session_name}")
        self._update_counts()

    @QtCore.Slot()
    def _stop_recording(self):
        """停止记录: 丢弃不完整 bin, 关闭文件"""
        self._recording = False

        # 关闭文件
        if self._raw_file:
            self._raw_file.close()
            self._raw_file = None
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        self.record_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.rec_status.setText(f"✅ 已保存: {self._session_name}")

        # 弹窗确认
        QtWidgets.QMessageBox.information(
            self, "保存完成",
            f"已保存:\n"
            f"  {self._session_name}_raw.bin  ({self._total_samples} 采样点)\n"
            f"  {self._session_name}_labels.csv  ({self._fist_bins + self._relax_bins} bins)\n\n"
            f"  Fist: {self._fist_bins}  Relax: {self._relax_bins}"
        )

        self.status_label.setText(f"已保存: {self._session_name}")

    # ── 波形 ────────────────────────────────────

    @QtCore.Slot()
    def _on_auto_y_changed(self):
        if self.auto_y_check.isChecked():
            self.plot_widget.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
            self.y_min_box.setEnabled(False)
            self.y_max_box.setEnabled(False)
        else:
            self.plot_widget.disableAutoRange(axis=pg.ViewBox.YAxis)
            self.y_min_box.setEnabled(True)
            self.y_max_box.setEnabled(True)
            self._on_manual_y_changed()

    @QtCore.Slot()
    def _on_manual_y_changed(self):
        if self.auto_y_check.isChecked():
            return
        y_min = self.y_min_box.value()
        y_max = self.y_max_box.value()
        if y_min < y_max:
            self.plot_widget.setYRange(y_min, y_max, padding=0)

    @QtCore.Slot()
    def _replay_saved(self):
        """选择已保存的 raw.bin + labels.csv, 在新窗口回放波形 (fist=红, relax=蓝)"""
        raw_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 raw.bin", str(Path(__file__).resolve().parent),
            "Raw Binary (*_raw.bin)")
        if not raw_path:
            return
        raw_path = Path(raw_path)
        csv_path = Path(str(raw_path).replace("_raw.bin", "_labels.csv"))
        if not csv_path.exists():
            QtWidgets.QMessageBox.warning(self, "缺少文件", f"未找到: {csv_path}")
            return

        # 读 raw.bin
        raw_data = np.fromfile(str(raw_path), dtype=np.int16)
        # 读 labels.csv
        labels = {}
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 4:
                    start, end, label = int(row[1]), int(row[2]), int(row[3])
                    labels[(start, end)] = label

        # 新窗口画图
        pw = pg.plot(title=f"回放: {raw_path.name} (红=fist, 蓝=relax)")
        pw.setBackground("w")
        pw.showGrid(x=True, y=True, alpha=0.3)
        pw.setLabel("bottom", "采样点")
        pw.setLabel("left", "ADC 值")
        pw.addLegend()

        # 波形 (灰色)
        x = np.arange(len(raw_data), dtype=float)
        y = raw_data.astype(float)
        pw.plot(x, y, pen=pg.mkPen(color="#cccccc", width=1), name="raw")

        # 分 bin 用对应颜色画
        for (start, end), label in labels.items():
            if start >= len(raw_data):
                continue
            end = min(end, len(raw_data) - 1)
            color = "#ff6666" if label == LABEL_FIST else "#6666ff"
            pw.plot(x[start:end+1], y[start:end+1],
                    pen=pg.mkPen(color=color, width=1.5))

    @QtCore.Slot()
    def _reset_view(self):
        """重置视图: 关闭自动范围, 恢复默认 Y 轴"""
        self.auto_y_check.setChecked(False)
        self.y_min_box.setValue(-500)
        self.y_max_box.setValue(2000)
        self.plot_widget.setYRange(-500, 2000)
        self.plot_widget.enableAutoRange(axis=pg.ViewBox.XAxis, enable=True)

    @QtCore.Slot()
    def _update_plot(self):
        n = min(self._plot_count, 5000)
        if n == 0:
            return
        if self._plot_count <= 5000:
            x = np.arange(n)
            y = self._plot_data[:n]
        else:
            x = np.arange(self._plot_count - 5000, self._plot_count)
            # self._plot_write 是下一个写入位置 = 最老数据起点
            w = self._plot_write
            y = np.concatenate([self._plot_data[w:], self._plot_data[:w]])
        self.curve.setData(x, y)

    def closeEvent(self, event):
        if self._recording:
            self._stop_recording()
        if self._connected:
            self._disconnect()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="EMG 数据采集工具")
    parser.add_argument("--port", help="串口号, 如 COM3")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("EMG Collect GUI")
    app.setStyle("Fusion")

    window = MainWindow(requested_port=args.port)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
