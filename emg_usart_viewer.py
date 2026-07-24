#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMG USART real-time reader and plotter.

Protocol derived from EMG-CH2.ini:
    Serial: 115200 baud, 8 data bits, no parity, 1 stop bit, no flow control
    Frame:  AA BB + 8-byte payload
    Payload: 4 x signed int16, little-endian
    Total frame length: 10 bytes
    Checksum: none

Default display offsets from the INI file:
    Channel 1: -1000
    Channel 2:     0
    Channel 3: -1000
    Channel 4:     0

Install:
    pip install pyserial numpy pyqtgraph PySide6

Run:
    python emg_usart_viewer.py
    python emg_usart_viewer.py --port COM3 --autoconnect
    python emg_usart_viewer.py --config EMG-CH2.ini
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import serial
from serial.tools import list_ports

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


@dataclass
class ProtocolConfig:
    # Serial settings
    selected_port: str = "COM3"
    baud_rate: int = 115200
    data_bits: int = 8
    parity: str = "none"
    stop_bits: float = 1
    flow_control: str = "none"

    # Frame settings
    header: bytes = b"\xAA\xBB"
    num_channels: int = 4
    number_format: str = "int16"
    endianness: str = "little"
    payload_size: int = 8
    checksum: bool = False

    # Channel/display settings
    channel_names: list[str] = field(
        default_factory=lambda: ["Channel1", "Channel2", "Channel 3", "Channel 4"]
    )
    channel_offsets: np.ndarray = field(
        default_factory=lambda: np.array([-1000.0, 0.0, -1000.0, 0.0])
    )
    channel_gains: np.ndarray = field(
        default_factory=lambda: np.ones(4, dtype=float)
    )
    channel_visible: list[bool] = field(
        default_factory=lambda: [True, True, True, True]
    )

    # Plot settings
    num_samples: int = 5000
    y_min: float = -500.0
    y_max: float = 500.0
    grid: bool = True
    legend: bool = True
    dark_background: bool = True

    @property
    def endian_prefix(self) -> str:
        if self.endianness.lower() == "little":
            return "<"
        if self.endianness.lower() == "big":
            return ">"
        raise ValueError(f"Unsupported endianness: {self.endianness}")

    @property
    def struct_code(self) -> str:
        formats = {
            "int16": "h",
            "uint16": "H",
        }
        try:
            return formats[self.number_format.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported number format: {self.number_format}. "
                "Supported formats: int16, uint16."
            ) from exc

    @property
    def sample_struct(self) -> struct.Struct:
        return struct.Struct(
            self.endian_prefix + self.struct_code * self.num_channels
        )

    @property
    def frame_size(self) -> int:
        return len(self.header) + self.payload_size

    def validate(self) -> None:
        expected_payload = self.sample_struct.size
        if expected_payload != self.payload_size:
            raise ValueError(
                "Protocol configuration is inconsistent: "
                f"{self.num_channels} × {self.number_format} requires "
                f"{expected_payload} payload bytes, but frameSize is "
                f"{self.payload_size}."
            )
        if self.checksum:
            raise ValueError(
                "This program currently expects checksum=false, matching "
                "EMG-CH2.ini."
            )
        if len(self.channel_names) != self.num_channels:
            raise ValueError("Channel-name count does not match num_channels.")
        if len(self.channel_offsets) != self.num_channels:
            raise ValueError("Offset count does not match num_channels.")
        if len(self.channel_gains) != self.num_channels:
            raise ValueError("Gain count does not match num_channels.")


def _get_bool(section: configparser.SectionProxy, key: str, default: bool) -> bool:
    try:
        return section.getboolean(key)
    except (ValueError, configparser.NoOptionError):
        return default


def load_ini(path: Optional[Path]) -> ProtocolConfig:
    """Load the relevant protocol and display fields from the Qt INI file."""
    cfg = ProtocolConfig()

    if path is None or not path.exists():
        cfg.validate()
        return cfg

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read(path, encoding="utf-8")

    if parser.has_section("Port"):
        sec = parser["Port"]
        cfg.selected_port = sec.get("selectedPort", cfg.selected_port)
        cfg.baud_rate = sec.getint("baudRate", cfg.baud_rate)
        cfg.parity = sec.get("parity", cfg.parity).lower()
        cfg.data_bits = sec.getint("dataBits", cfg.data_bits)
        cfg.stop_bits = sec.getfloat("stopBits", cfg.stop_bits)
        cfg.flow_control = sec.get("flowControl", cfg.flow_control).lower()

    if parser.has_section("DataFormat_CustomFrame"):
        sec = parser["DataFormat_CustomFrame"]
        cfg.num_channels = sec.getint("numOfChannels", cfg.num_channels)
        cfg.number_format = sec.get("numberFormat", cfg.number_format).lower()
        cfg.endianness = sec.get("endianness", cfg.endianness).lower()
        frame_start = sec.get("frameStart", "AA BB")
        cfg.header = bytes.fromhex(frame_start)
        cfg.payload_size = sec.getint("frameSize", cfg.payload_size)
        cfg.checksum = _get_bool(sec, "checksum", cfg.checksum)

    cfg.channel_names = []
    offsets = []
    gains = []
    visible = []

    if parser.has_section("Channels"):
        sec = parser["Channels"]
        for channel_index in range(1, cfg.num_channels + 1):
            prefix = f"channel\\{channel_index}\\"
            cfg.channel_names.append(
                sec.get(prefix + "name", f"Channel {channel_index}")
            )

            offset_enabled = _get_bool(
                sec, prefix + "offsetEnabled", False
            )
            offset = sec.getfloat(prefix + "offset", 0.0)
            offsets.append(offset if offset_enabled else 0.0)

            gain_enabled = _get_bool(
                sec, prefix + "gainEnabled", False
            )
            gain = sec.getfloat(prefix + "gain", 1.0)
            gains.append(gain if gain_enabled else 1.0)

            visible.append(
                _get_bool(sec, prefix + "visible", True)
            )
    else:
        cfg.channel_names = [
            f"Channel {i}" for i in range(1, cfg.num_channels + 1)
        ]
        offsets = [0.0] * cfg.num_channels
        gains = [1.0] * cfg.num_channels
        visible = [True] * cfg.num_channels

    cfg.channel_offsets = np.asarray(offsets, dtype=float)
    cfg.channel_gains = np.asarray(gains, dtype=float)
    cfg.channel_visible = visible

    if parser.has_section("Plot"):
        sec = parser["Plot"]
        cfg.num_samples = sec.getint("numOfSamples", cfg.num_samples)
        cfg.y_min = sec.getfloat("yMin", cfg.y_min)
        cfg.y_max = sec.getfloat("yMax", cfg.y_max)
        cfg.grid = _get_bool(sec, "grid", cfg.grid)
        cfg.legend = _get_bool(sec, "legend", cfg.legend)
        cfg.dark_background = _get_bool(
            sec, "darkBackground", cfg.dark_background
        )

    cfg.validate()
    return cfg


class FrameParser:
    """
    Streaming fixed-length frame parser.

    It searches for the header, discards bytes before it, decodes one complete
    fixed-size frame, and then searches for the next header again. Therefore it
    automatically recovers after startup in the middle of a frame or after a
    dropped/extra byte.
    """

    def __init__(self, config: ProtocolConfig):
        self.config = config
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.frames_parsed = 0

    def reset(self) -> None:
        self.buffer.clear()
        self.discarded_bytes = 0
        self.frames_parsed = 0

    def feed(self, data: bytes) -> np.ndarray:
        if not data:
            return np.empty(
                (0, self.config.num_channels), dtype=np.int16
            )

        self.buffer.extend(data)
        decoded: list[tuple[int, ...]] = []

        while True:
            header_index = self.buffer.find(self.config.header)

            if header_index < 0:
                # Preserve a possible partial header at the end.
                keep = max(0, len(self.config.header) - 1)
                if keep and len(self.buffer) >= keep:
                    tail = self.buffer[-keep:]
                    discarded = len(self.buffer) - keep
                    self.discarded_bytes += discarded
                    self.buffer[:] = tail
                else:
                    self.discarded_bytes += len(self.buffer)
                    self.buffer.clear()
                break

            if header_index > 0:
                self.discarded_bytes += header_index
                del self.buffer[:header_index]

            if len(self.buffer) < self.config.frame_size:
                break

            payload_start = len(self.config.header)
            payload_end = payload_start + self.config.payload_size
            payload = bytes(self.buffer[payload_start:payload_end])

            try:
                values = self.config.sample_struct.unpack(payload)
            except struct.error:
                # Defensive fallback: discard one byte and search again.
                self.discarded_bytes += 1
                del self.buffer[0]
                continue

            decoded.append(values)
            self.frames_parsed += 1
            del self.buffer[:self.config.frame_size]

        if not decoded:
            dtype = (
                np.int16
                if self.config.number_format == "int16"
                else np.uint16
            )
            return np.empty((0, self.config.num_channels), dtype=dtype)

        dtype = (
            np.int16
            if self.config.number_format == "int16"
            else np.uint16
        )
        return np.asarray(decoded, dtype=dtype)


class SerialReader(QtCore.QThread):
    samples_received = QtCore.Signal(object, float)
    connected = QtCore.Signal(str)
    disconnected = QtCore.Signal()
    error = QtCore.Signal(str)
    parser_stats = QtCore.Signal(int, int)

    def __init__(self, port: str, baud: int, config: ProtocolConfig):
        super().__init__()
        self.port = port
        self.baud = baud
        self.config = config
        self._running = False
        self._serial: Optional[serial.Serial] = None
        self.parser = FrameParser(config)

    def _serial_parameters(self) -> dict:
        byte_sizes = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS,
        }
        parities = {
            "none": serial.PARITY_NONE,
            "even": serial.PARITY_EVEN,
            "odd": serial.PARITY_ODD,
            "mark": serial.PARITY_MARK,
            "space": serial.PARITY_SPACE,
        }
        stop_bits = {
            1.0: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2.0: serial.STOPBITS_TWO,
        }

        flow = self.config.flow_control.lower()
        return {
            "bytesize": byte_sizes[self.config.data_bits],
            "parity": parities[self.config.parity],
            "stopbits": stop_bits[float(self.config.stop_bits)],
            "xonxoff": flow in {"software", "xonxoff", "xon/xoff"},
            "rtscts": flow in {"hardware", "rtscts", "rts/cts"},
            "dsrdtr": flow in {"dsrdtr", "dsr/dtr"},
        }

    def run(self) -> None:
        self._running = True
        self.parser.reset()

        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.05,
                write_timeout=0.5,
                **self._serial_parameters(),
            )
            self._serial.reset_input_buffer()
            self.connected.emit(self.port)

            last_stats_time = time.monotonic()

            while self._running:
                waiting = self._serial.in_waiting
                chunk = self._serial.read(waiting if waiting > 0 else 1)

                if chunk:
                    frames = self.parser.feed(chunk)
                    if frames.shape[0]:
                        self.samples_received.emit(frames, time.time())

                now = time.monotonic()
                if now - last_stats_time >= 0.5:
                    self.parser_stats.emit(
                        self.parser.frames_parsed,
                        self.parser.discarded_bytes,
                    )
                    last_stats_time = now

        except (serial.SerialException, OSError, KeyError, ValueError) as exc:
            if self._running:
                self.error.emit(str(exc))
        finally:
            self._running = False
            if self._serial is not None:
                try:
                    self._serial.close()
                except serial.SerialException:
                    pass
                self._serial = None
            self.disconnected.emit()

    def stop(self) -> None:
        self._running = False
        if self._serial is not None:
            try:
                self._serial.cancel_read()
            except (AttributeError, serial.SerialException):
                pass


class CircularSampleBuffer:
    def __init__(self, channels: int, capacity: int):
        self.channels = channels
        self.capacity = capacity
        self.data = np.zeros((channels, capacity), dtype=float)
        self.write_index = 0
        self.count = 0

    def clear(self) -> None:
        self.data.fill(0.0)
        self.write_index = 0
        self.count = 0

    def append(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return

        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError("Unexpected sample-array shape.")

        if samples.shape[0] >= self.capacity:
            latest = samples[-self.capacity :]
            self.data[:, :] = latest.T
            self.write_index = 0
            self.count = self.capacity
            return

        number = samples.shape[0]
        first_part = min(number, self.capacity - self.write_index)
        self.data[
            :, self.write_index : self.write_index + first_part
        ] = samples[:first_part].T

        remaining = number - first_part
        if remaining:
            self.data[:, :remaining] = samples[first_part:].T

        self.write_index = (self.write_index + number) % self.capacity
        self.count = min(self.capacity, self.count + number)

    def chronological(self) -> np.ndarray:
        if self.count == 0:
            return np.empty((self.channels, 0), dtype=float)

        if self.count < self.capacity:
            return self.data[:, : self.count].copy()

        return np.concatenate(
            (
                self.data[:, self.write_index :],
                self.data[:, : self.write_index],
            ),
            axis=1,
        )


class MainWindow(QtWidgets.QMainWindow):
    CURVE_COLORS = [
        "#ff5656",
        "#ff7e2d",
        "#00ae7e",
        "#00b917",
        "#ff937e",
        "#00aaff",
        "#ff029d",
        "#7a4782",
    ]

    def __init__(
        self,
        config: ProtocolConfig,
        requested_port: Optional[str] = None,
        requested_baud: Optional[int] = None,
        autoconnect: bool = False,
        apply_offsets: bool = True,
    ):
        super().__init__()
        self.config = config
        self.reader: Optional[SerialReader] = None
        self.sample_buffer = CircularSampleBuffer(
            config.num_channels, config.num_samples
        )
        self.total_samples = 0
        self.current_rate = 0.0
        self.last_rate_count = 0
        self.last_rate_time = time.monotonic()
        self.last_raw = np.zeros(config.num_channels, dtype=float)

        self.csv_file = None
        self.csv_writer = None
        self.record_rows_since_flush = 0
        self.session_start_monotonic = time.monotonic()

        # Recording state
        self.recording_mode: Optional[str] = None  # None | "bend" | "relax"
        self.recording_elapsed: float = 0.0
        self.recording_sample_count: int = 0
        self.target_samples: int = 2500  # 5 s × 500 samples/s

        # Auto-save: counter naming + save directory
        script_dir = Path(__file__).resolve().parent
        self.save_dir = script_dir.parent / "data_saved"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.bend_counter, self.relax_counter = self._scan_counters()
        self.current_action: str = ""  # "bend" | "relax" | ""

        self.setWindowTitle("EMG USART Viewer — AA BB + 4×int16 LE")
        self.resize(1200, 820)

        self._build_ui(apply_offsets)
        self.refresh_ports()

        initial_port = requested_port or config.selected_port
        self.port_box.setCurrentText(initial_port)
        self.baud_box.setCurrentText(
            str(requested_baud or config.baud_rate)
        )

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setInterval(33)
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start()

        self.rate_timer = QtCore.QTimer(self)
        self.rate_timer.setInterval(1000)
        self.rate_timer.timeout.connect(self.update_rate)
        self.rate_timer.start()

        self.recording_timer = QtCore.QTimer(self)
        self.recording_timer.setInterval(100)
        self.recording_timer.timeout.connect(self.update_recording_display)

        if autoconnect:
            QtCore.QTimer.singleShot(200, self.toggle_connection)

    def _build_ui(self, apply_offsets: bool) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QGridLayout()
        main_layout.addLayout(controls)

        controls.addWidget(QtWidgets.QLabel("Serial port:"), 0, 0)
        self.port_box = QtWidgets.QComboBox()
        self.port_box.setEditable(True)
        self.port_box.setMinimumWidth(130)
        controls.addWidget(self.port_box, 0, 1)

        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_ports)
        controls.addWidget(self.refresh_button, 0, 2)

        controls.addWidget(QtWidgets.QLabel("Baud:"), 0, 3)
        self.baud_box = QtWidgets.QComboBox()
        self.baud_box.setEditable(True)
        self.baud_box.addItems(
            ["9600", "19200", "38400", "57600", "115200", "230400", "460800"]
        )
        controls.addWidget(self.baud_box, 0, 4)

        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.setMinimumWidth(110)
        self.connect_button.clicked.connect(self.toggle_connection)
        controls.addWidget(self.connect_button, 0, 5)

        self.record_bend_button = QtWidgets.QPushButton("💪 弯曲 5s")
        self.record_bend_button.setFixedWidth(125)
        self.record_bend_button.clicked.connect(
            self.start_bend_recording
        )
        controls.addWidget(self.record_bend_button, 0, 6)

        self.record_relax_button = QtWidgets.QPushButton("🖐 伸直 5s")
        self.record_relax_button.setFixedWidth(125)
        self.record_relax_button.clicked.connect(
            self.start_relax_recording
        )
        controls.addWidget(self.record_relax_button, 0, 7)

        self.clear_button = QtWidgets.QPushButton("Clear plot")
        self.clear_button.clicked.connect(self.clear_data)
        controls.addWidget(self.clear_button, 0, 8)

        self.offset_checkbox = QtWidgets.QCheckBox("Apply INI offsets/gains")
        self.offset_checkbox.setChecked(apply_offsets)
        self.offset_checkbox.stateChanged.connect(self.refresh_value_labels)
        controls.addWidget(self.offset_checkbox, 1, 0, 1, 2)

        self.autoscale_checkbox = QtWidgets.QCheckBox("Auto-scale Y")
        self.autoscale_checkbox.stateChanged.connect(self.on_autoscale_changed)
        controls.addWidget(self.autoscale_checkbox, 1, 2)

        controls.addWidget(QtWidgets.QLabel("Y min:"), 1, 3)
        self.y_min_box = QtWidgets.QDoubleSpinBox()
        self.y_min_box.setRange(-32768, 32767)
        self.y_min_box.setDecimals(0)
        self.y_min_box.setValue(self.config.y_min)
        self.y_min_box.valueChanged.connect(self.on_manual_y_changed)
        controls.addWidget(self.y_min_box, 1, 4)

        controls.addWidget(QtWidgets.QLabel("Y max:"), 1, 5)
        self.y_max_box = QtWidgets.QDoubleSpinBox()
        self.y_max_box.setRange(-32768, 32767)
        self.y_max_box.setDecimals(0)
        self.y_max_box.setValue(self.config.y_max)
        self.y_max_box.valueChanged.connect(self.on_manual_y_changed)
        controls.addWidget(self.y_max_box, 1, 6)

        self.recording_duration_label = QtWidgets.QLabel("")
        self.recording_duration_label.setMinimumWidth(180)
        self.recording_duration_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        font = self.recording_duration_label.font()
        font.setBold(True)
        self.recording_duration_label.setFont(font)
        controls.addWidget(self.recording_duration_label, 1, 7)

        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        controls.addWidget(self.status_label, 1, 8)

        # Large action indicator label
        self.action_label = QtWidgets.QLabel("")
        self.action_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.action_label.setMinimumHeight(55)
        action_font = self.action_label.font()
        action_font.setPointSize(28)
        action_font.setBold(True)
        self.action_label.setFont(action_font)
        main_layout.addWidget(self.action_label)

        channel_group = QtWidgets.QGroupBox("Channels")
        channel_layout = QtWidgets.QGridLayout(channel_group)
        main_layout.addWidget(channel_group)

        self.channel_checkboxes = []
        self.raw_value_labels = []
        self.display_value_labels = []

        for i in range(self.config.num_channels):
            checkbox = QtWidgets.QCheckBox(self.config.channel_names[i])
            checkbox.setChecked(self.config.channel_visible[i])
            checkbox.stateChanged.connect(self.update_curve_visibility)
            self.channel_checkboxes.append(checkbox)

            raw_label = QtWidgets.QLabel("raw: —")
            display_label = QtWidgets.QLabel("display: —")
            self.raw_value_labels.append(raw_label)
            self.display_value_labels.append(display_label)

            column = i % 4
            row = (i // 4) * 2
            channel_layout.addWidget(checkbox, row, column)
            values_widget = QtWidgets.QWidget()
            values_layout = QtWidgets.QHBoxLayout(values_widget)
            values_layout.setContentsMargins(18, 0, 6, 0)
            values_layout.addWidget(raw_label)
            values_layout.addWidget(display_label)
            channel_layout.addWidget(values_widget, row + 1, column)

        pg.setConfigOptions(antialias=False)
        self.plot = pg.PlotWidget()
        if self.config.dark_background:
            self.plot.setBackground("k")
        self.plot.showGrid(
            x=self.config.grid,
            y=self.config.grid,
            alpha=0.25,
        )
        self.plot.setLabel("bottom", "Sample index")
        self.plot.setLabel("left", "Amplitude")
        self.plot.setYRange(self.config.y_min, self.config.y_max)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode="peak")
        main_layout.addWidget(self.plot, stretch=1)

        if self.config.legend:
            self.legend = self.plot.addLegend()
        else:
            self.legend = None

        self.curves = []
        for i in range(self.config.num_channels):
            color = self.CURVE_COLORS[i % len(self.CURVE_COLORS)]
            curve = self.plot.plot(
                pen=pg.mkPen(color=color, width=1.2),
                name=self.config.channel_names[i],
            )
            curve.setVisible(self.config.channel_visible[i])
            self.curves.append(curve)

        self.protocol_label = QtWidgets.QLabel(
            "Protocol: header AA BB | payload 8 bytes | "
            "4 × signed int16 little-endian | no checksum | total 10 bytes"
        )
        self.protocol_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        main_layout.addWidget(self.protocol_label)

    def processed_samples(self, raw: np.ndarray) -> np.ndarray:
        raw_float = np.asarray(raw, dtype=float)
        if not self.offset_checkbox.isChecked():
            return raw_float
        return (
            raw_float * self.config.channel_gains
            + self.config.channel_offsets
        )

    @QtCore.Slot()
    def refresh_ports(self) -> None:
        current = self.port_box.currentText().strip()
        ports = sorted(port.device for port in list_ports.comports())

        self.port_box.blockSignals(True)
        self.port_box.clear()
        self.port_box.addItems(ports)
        self.port_box.setCurrentText(current or self.config.selected_port)
        self.port_box.blockSignals(False)

    @QtCore.Slot()
    def toggle_connection(self) -> None:
        if self.reader is not None and self.reader.isRunning():
            self.disconnect_serial()
            return

        port = self.port_box.currentText().strip()
        if not port:
            QtWidgets.QMessageBox.warning(
                self, "Missing serial port", "Select or type a serial port."
            )
            return

        try:
            baud = int(self.baud_box.currentText())
            if baud <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid baud rate", "Baud rate must be a positive integer."
            )
            return

        self.connect_button.setEnabled(False)
        self.status_label.setText(f"Opening {port}…")
        self.session_start_monotonic = time.monotonic()
        self.last_rate_time = self.session_start_monotonic
        self.last_rate_count = self.total_samples

        self.reader = SerialReader(port, baud, self.config)
        self.reader.samples_received.connect(self.on_samples_received)
        self.reader.connected.connect(self.on_connected)
        self.reader.disconnected.connect(self.on_disconnected)
        self.reader.error.connect(self.on_reader_error)
        self.reader.parser_stats.connect(self.on_parser_stats)
        self.reader.start()

    def disconnect_serial(self) -> None:
        if self.reader is not None:
            self.status_label.setText("Disconnecting…")
            self.connect_button.setEnabled(False)
            self.reader.stop()
            self.reader.wait(1500)

    @QtCore.Slot(str)
    def on_connected(self, port: str) -> None:
        self.connect_button.setText("Disconnect")
        self.connect_button.setEnabled(True)
        self.port_box.setEnabled(False)
        self.baud_box.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.status_label.setText(f"Connected: {port}")

    @QtCore.Slot()
    def on_disconnected(self) -> None:
        self.connect_button.setText("Connect")
        self.connect_button.setEnabled(True)
        self.port_box.setEnabled(True)
        self.baud_box.setEnabled(True)
        self.refresh_button.setEnabled(True)

        if not self.status_label.text().startswith("Error"):
            self.status_label.setText("Disconnected")

        if self.reader is not None:
            self.reader.deleteLater()
            self.reader = None

    @QtCore.Slot(str)
    def on_reader_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        QtWidgets.QMessageBox.critical(
            self, "Serial-port error", message
        )

    @QtCore.Slot(object, float)
    def on_samples_received(
        self, raw_samples: np.ndarray, receive_time: float
    ) -> None:
        if raw_samples.size == 0:
            return

        raw_samples = np.asarray(raw_samples)
        self.sample_buffer.append(raw_samples)
        self.last_raw = raw_samples[-1].astype(float)
        first_frame_index = self.total_samples
        self.total_samples += raw_samples.shape[0]

        self.refresh_value_labels()

        if self.csv_writer is not None:
            try:
                display_samples = self.processed_samples(raw_samples)
                elapsed = time.monotonic() - self.session_start_monotonic
                iso_time = datetime.fromtimestamp(
                    receive_time
                ).astimezone().isoformat(timespec="milliseconds")

                for row_index, (raw_row, shown_row) in enumerate(
                    zip(raw_samples, display_samples)
                ):
                    self.csv_writer.writerow(
                        [
                            first_frame_index + row_index,
                            iso_time,
                            f"{elapsed:.6f}",
                            *[int(v) for v in raw_row],
                            *[f"{float(v):.6f}" for v in shown_row],
                        ]
                    )

                self.record_rows_since_flush += raw_samples.shape[0]
                if self.record_rows_since_flush >= 500:
                    self.csv_file.flush()
                    self.record_rows_since_flush = 0

                # Track samples for duration display and auto-stop
                self.recording_sample_count += raw_samples.shape[0]
                if self.recording_sample_count >= self.target_samples:
                    self.stop_recording()
            except (OSError, ValueError, OverflowError) as exc:
                print(
                    f"CSV write error: {exc}",
                    file=sys.stderr,
                )

    @QtCore.Slot(int, int)
    def on_parser_stats(self, frames: int, discarded: int) -> None:
        connection = (
            self.reader.port
            if self.reader is not None
            else "serial"
        )
        self.status_label.setText(
            f"{connection} | {self.current_rate:.1f} frames/s | "
            f"total {self.total_samples:,} | discarded {discarded} bytes"
        )

    def refresh_value_labels(self) -> None:
        displayed = self.processed_samples(self.last_raw[None, :])[0]
        for i in range(self.config.num_channels):
            self.raw_value_labels[i].setText(
                f"raw: {int(self.last_raw[i])}"
            )
            self.display_value_labels[i].setText(
                f"display: {displayed[i]:.0f}"
            )

    @QtCore.Slot()
    def update_rate(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_rate_time
        if elapsed > 0:
            count_delta = self.total_samples - self.last_rate_count
            self.current_rate = count_delta / elapsed
        self.last_rate_count = self.total_samples
        self.last_rate_time = now

    @QtCore.Slot()
    def update_plot(self) -> None:
        raw = self.sample_buffer.chronological()
        if raw.shape[1] == 0:
            return

        displayed = self.processed_samples(raw.T).T
        x_start = self.total_samples - displayed.shape[1]
        x = np.arange(x_start, self.total_samples, dtype=float)

        for channel, curve in enumerate(self.curves):
            if self.channel_checkboxes[channel].isChecked():
                curve.setData(x, displayed[channel])

        if self.autoscale_checkbox.isChecked():
            self.plot.enableAutoRange(
                axis=pg.ViewBox.YAxis, enable=True
            )
        else:
            self.plot.disableAutoRange(axis=pg.ViewBox.YAxis)
            self.plot.setYRange(
                self.y_min_box.value(),
                self.y_max_box.value(),
                padding=0,
            )

    @QtCore.Slot()
    def update_curve_visibility(self) -> None:
        for checkbox, curve in zip(
            self.channel_checkboxes, self.curves
        ):
            curve.setVisible(checkbox.isChecked())

    @QtCore.Slot()
    def on_autoscale_changed(self) -> None:
        enabled = self.autoscale_checkbox.isChecked()
        self.y_min_box.setEnabled(not enabled)
        self.y_max_box.setEnabled(not enabled)

        if enabled:
            self.plot.enableAutoRange(
                axis=pg.ViewBox.YAxis, enable=True
            )
        else:
            self.on_manual_y_changed()

    @QtCore.Slot()
    def on_manual_y_changed(self) -> None:
        if self.autoscale_checkbox.isChecked():
            return

        y_min = self.y_min_box.value()
        y_max = self.y_max_box.value()
        if y_min >= y_max:
            return

        self.plot.setYRange(y_min, y_max, padding=0)

    @QtCore.Slot()
    def clear_data(self) -> None:
        self.sample_buffer.clear()
        self.total_samples = 0
        self.last_raw.fill(0.0)
        self.refresh_value_labels()
        for curve in self.curves:
            curve.clear()

    # ── Recording: auto-save helpers ──────────────────────────

    def _scan_counters(self) -> tuple[int, int]:
        """Scan save_dir for existing files and return (bend_max, relax_max)."""
        bend_max = 0
        relax_max = 0
        if self.save_dir.exists():
            for f in self.save_dir.glob("bend_*.csv"):
                try:
                    n = int(f.stem.split("_")[1])
                    bend_max = max(bend_max, n)
                except (ValueError, IndexError):
                    pass
            for f in self.save_dir.glob("relax_*.csv"):
                try:
                    n = int(f.stem.split("_")[1])
                    relax_max = max(relax_max, n)
                except (ValueError, IndexError):
                    pass
        return bend_max, relax_max

    def _next_filename(self, action: str) -> Path:
        """Generate the next counter-based filename and increment the counter."""
        if action == "bend":
            self.bend_counter += 1
            return self.save_dir / f"bend_{self.bend_counter:03d}.csv"
        else:
            self.relax_counter += 1
            return self.save_dir / f"relax_{self.relax_counter:03d}.csv"

    def _start_recording(self, action: str) -> bool:
        """Open CSV via auto-save and write header.  Return True on success."""
        filepath = self._next_filename(action)
        try:
            self.csv_file = open(
                str(filepath), "w", newline="", encoding="utf-8"
            )
            self.csv_writer = csv.writer(self.csv_file)

            raw_headers = [
                f"{name}_raw" for name in self.config.channel_names
            ]
            display_headers = [
                f"{name}_display" for name in self.config.channel_names
            ]
            self.csv_writer.writerow(
                [
                    "frame_index",
                    "host_receive_time",
                    "elapsed_s",
                    *raw_headers,
                    *display_headers,
                ]
            )
            self.csv_file.flush()
            self.record_rows_since_flush = 0
            return True
        except OSError as exc:
            self.csv_file = None
            self.csv_writer = None
            QtWidgets.QMessageBox.critical(
                self, "Cannot create CSV file", str(exc)
            )
            return False

    def _recording_started(self, action: str) -> None:
        """Common state update when recording begins."""
        self.recording_mode = action
        self.recording_elapsed = 0.0
        self.recording_sample_count = 0
        self.current_action = action
        self.recording_timer.start()
        self._update_recording_ui_state()

    def stop_recording(self) -> None:
        """Stop any active recording and close the CSV file."""
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except OSError:
                pass

        self.csv_file = None
        self.csv_writer = None
        self.recording_mode = None
        self.current_action = ""
        self.recording_timer.stop()
        self._update_recording_ui_state()

    def _update_recording_ui_state(self) -> None:
        """Enable/disable buttons and labels according to recording mode."""
        recording = self.recording_mode is not None
        self.record_bend_button.setEnabled(not recording)
        self.record_relax_button.setEnabled(not recording)

        if not recording:
            self.recording_duration_label.setText("")
            self.recording_duration_label.setStyleSheet("")
            self.action_label.setText("")
            self.action_label.setStyleSheet("")

    @QtCore.Slot()
    def update_recording_display(self) -> None:
        """Called by recording_timer every 100 ms."""
        if self.recording_mode is None:
            return

        # Accurate elapsed time from sample count (500 samples/s)
        self.recording_elapsed = self.recording_sample_count / 500.0
        minutes = int(self.recording_elapsed // 60)
        seconds = self.recording_elapsed % 60

        text = (
            f"⏺ {minutes:01d}:{seconds:04.1f} │ "
            f"{self.recording_sample_count}/{self.target_samples}"
        )
        self.recording_duration_label.setText(text)
        self.recording_duration_label.setStyleSheet(
            "color: #ff4444; font-weight: bold;"
        )

        # Large action indicator
        if self.recording_mode == "bend":
            self.action_label.setText("💪  弯  曲  💪")
            self.action_label.setStyleSheet(
                "color: #ff4444; font-weight: bold;"
            )
        else:
            self.action_label.setText("🖐  伸  直  🖐")
            self.action_label.setStyleSheet(
                "color: #44aaff; font-weight: bold;"
            )

    # ── Recording: button callbacks ───────────────────────────

    @QtCore.Slot()
    def start_bend_recording(self) -> None:
        """Start a 5-second bend recording."""
        if self.recording_mode is not None:
            return
        if not self._start_recording("bend"):
            return
        self._recording_started("bend")

    @QtCore.Slot()
    def start_relax_recording(self) -> None:
        """Start a 5-second relax recording."""
        if self.recording_mode is not None:
            return
        if not self._start_recording("relax"):
            return
        self._recording_started("relax")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_recording()
        if self.reader is not None and self.reader.isRunning():
            self.reader.stop()
            self.reader.wait(1500)
        event.accept()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read and display EMG USART frames: "
            "AA BB + 4×int16 little-endian."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("EMG-CH2.ini"),
        help=(
            "Path to EMG-CH2.ini. Defaults are used when the file "
            "does not exist."
        ),
    )
    parser.add_argument(
        "--port",
        help="Serial port, for example COM3 or /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        help="Override the baud rate from the INI file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Number of samples retained in the live plot.",
    )
    parser.add_argument(
        "--autoconnect",
        action="store_true",
        help="Open the selected serial port after the GUI starts.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Start with INI offsets/gains disabled.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    try:
        config = load_ini(args.config)
        if args.samples is not None:
            if args.samples <= 0:
                raise ValueError("--samples must be positive.")
            config.num_samples = args.samples
        config.validate()
    except (OSError, ValueError, configparser.Error) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("EMG USART Viewer")
    app.setStyle("Fusion")

    window = MainWindow(
        config=config,
        requested_port=args.port,
        requested_baud=args.baud,
        autoconnect=args.autoconnect,
        apply_offsets=not args.raw,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
