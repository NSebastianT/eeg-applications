"""
Muse 2 signal diagnostic and manual event labeling tool.

This script has two modes:

1. Diagnostic mode:
   - Live visualization of Muse 2 EEG channels exposed through BrainFlow.
   - No motor control.
   - No serial communication.
   - No data saving.

2. Save mode (--save):
   - Live visualization.
   - Manual segment labeling using the keyboard.
   - Saves local EEG data under data/raw/manual_events/.
   - Saved data is biometric/local data and should not be pushed to Git.

Keyboard controls in --save mode:
    0-9   Select label
    R     Start/stop current segment
    U     Undo last completed segment
    ESC   Cancel current open segment
    Q     Save summary and quit

Keyboard controls in diagnostic mode:
    Q     Quit

This script does not classify events by itself. It creates a clean labeled
dataset for later feature extraction and classifier design.
"""

import argparse
import csv
import json
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


SCRIPT_NAME = "muse2_signal_diagnostic_v2.py"
SCRIPT_VERSION = "0.2.0"
BOARD_ID = BoardIds.MUSE_2_BOARD.value

CHANNEL_NAMES = ["TP9", "AF7", "AF8", "TP10"]

LABELS = {
    "0": "rest",
    "1": "blink_normal",
    "2": "blink_strong",
    "3": "blink_double",
    "4": "look_left_center",
    "5": "look_right_center",
    "6": "jaw_clench_release",
    "7": "eyebrow_fast",
    "8": "eyebrow_hold",
    "9": "headband_shift",
}

RAW_COLUMNS = [
    "sample_index",
    "time_s",
    "package_num",
    "TP9",
    "AF7",
    "AF8",
    "TP10",
    "other",
    "timestamp",
    "marker",
]

EVENT_COLUMNS = [
    "event_id",
    "label_key",
    "label",
    "label_trial_index",
    "start_sample",
    "end_sample",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "n_samples",
    "valid",
    "rejected_reason",
    "frontal_common_p2p_uv",
    "frontal_diff_p2p_uv",
    "temporal_common_p2p_uv",
    "frontal_step_uv_per_sample",
    "af7_af8_corr",
    "max_abs_raw_uv",
    "suspicious_extreme",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Muse 2 live diagnostic viewer and optional manual event labeler."
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Enable manual event labeling and save local data under data/raw/manual_events/.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw/manual_events",
        help="Root output directory used only in --save mode.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=6.0,
        help="Seconds shown in the live signal window.",
    )
    parser.add_argument(
        "--feature-window-seconds",
        type=float,
        default=0.50,
        help="Recent window used to compute diagnostic features.",
    )
    parser.add_argument(
        "--feature-history-seconds",
        type=float,
        default=60.0,
        help="Seconds shown in the feature history plot.",
    )
    parser.add_argument(
        "--update-ms",
        type=int,
        default=50,
        help="Plot update interval in milliseconds.",
    )
    parser.add_argument(
        "--frontal-threshold-uv",
        type=float,
        default=900.0,
        help="Diagnostic threshold for frontal common peak-to-peak activity.",
    )
    parser.add_argument(
        "--diff-threshold-uv",
        type=float,
        default=500.0,
        help="Diagnostic threshold for AF7-AF8 peak-to-peak activity.",
    )
    parser.add_argument(
        "--temporal-threshold-uv",
        type=float,
        default=700.0,
        help="Diagnostic threshold for temporal common peak-to-peak activity.",
    )
    parser.add_argument(
        "--reset-threshold-uv",
        type=float,
        default=450.0,
        help="Feature level used only for diagnostic state display.",
    )
    parser.add_argument(
        "--fast-change-threshold-uv",
        type=float,
        default=250.0,
        help="Diagnostic threshold for max sample-to-sample change.",
    )
    parser.add_argument(
        "--suspicious-extreme-uv",
        type=float,
        default=990.0,
        help="Approximate suspicious extreme-value threshold for live diagnostics.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=1.0,
        help="Console event cooldown after a detected candidate.",
    )
    parser.add_argument(
        "--mac-address",
        type=str,
        default="",
        help="Optional Muse 2 BLE MAC address. Leave empty for auto discovery.",
    )
    parser.add_argument(
        "--serial-number",
        type=str,
        default="",
        help="Optional Muse 2 serial number. Leave empty for auto discovery.",
    )
    parser.add_argument(
        "--brainflow-logger",
        action="store_true",
        help="Enable BrainFlow development logger.",
    )

    return parser.parse_args()


def peak_to_peak(x):
    if len(x) == 0:
        return 0.0
    return float(np.ptp(x))


def max_abs_step(x):
    if len(x) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(x))))


def safe_corr(a, b):
    if len(a) < 3:
        return np.nan

    a_std = np.std(a)
    b_std = np.std(b)

    if a_std < 1e-9 or b_std < 1e-9:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def candidate_label_from_features(
    frontal_common_p2p,
    frontal_diff_p2p,
    temporal_common_p2p,
    frontal_step,
    af_corr,
    args,
):
    labels = []

    frontal_event = frontal_common_p2p >= args.frontal_threshold_uv
    diff_event = frontal_diff_p2p >= args.diff_threshold_uv
    temporal_event = temporal_common_p2p >= args.temporal_threshold_uv
    fast_event = frontal_step >= args.fast_change_threshold_uv

    if frontal_event and fast_event and (np.isnan(af_corr) or af_corr >= 0.25):
        labels.append("front-fast candidate")

    if frontal_event and not fast_event:
        labels.append("front-sustained candidate")

    if diff_event and (np.isnan(af_corr) or af_corr <= 0.10):
        labels.append("AF7-AF8 asymmetric candidate")

    if temporal_event and temporal_common_p2p >= 0.65 * max(frontal_common_p2p, 1.0):
        labels.append("temporal-dominant candidate")

    if not labels and (frontal_event or diff_event or temporal_event):
        labels.append("unclassified high-amplitude candidate")

    return " + ".join(labels)


class MuseSignalDiagnosticViewer:
    def __init__(self, args):
        self.args = args
        self.cleaned_up = False

        if args.brainflow_logger:
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        self.params = BrainFlowInputParams()
        self.params.mac_address = args.mac_address
        self.params.serial_number = args.serial_number

        self.board = BoardShim(BOARD_ID, self.params)
        self.sampling_rate = BoardShim.get_sampling_rate(BOARD_ID)
        self.descr = BoardShim.get_board_descr(BOARD_ID)
        self.channel_map = self._get_channel_map()

        self.package_num_channel = self.descr.get("package_num_channel")
        self.timestamp_channel = self.descr.get("timestamp_channel")
        self.marker_channel = self.descr.get("marker_channel")
        self.other_channels = self.descr.get("other_channels", [])

        self.window_samples = int(args.window_seconds * self.sampling_rate)
        self.feature_samples = int(args.feature_window_seconds * self.sampling_rate)
        self.feature_samples = max(self.feature_samples, 8)

        self.display_buffers = {
            name: deque(maxlen=self.window_samples) for name in CHANNEL_NAMES
        }
        self.display_times = deque(maxlen=self.window_samples)

        self.feature_times = deque()
        self.frontal_history = deque()
        self.diff_history = deque()
        self.temporal_history = deque()

        self.total_samples = 0
        self.raw_rows = []

        self.selected_label_key = "1"
        self.selected_label = LABELS[self.selected_label_key]
        self.recording_segment = False
        self.segment_start_sample = None
        self.segment_start_time_s = None
        self.events = []
        self.event_counter = 0
        self.last_saved_message = "None"
        self.last_console_event_time = 0.0

        self.session_dir = None
        self.raw_file_handle = None
        self.raw_writer = None

        if self.args.save:
            self._setup_session_files()

        self.app = pg.mkQApp("Muse 2 Signal Diagnostic v2")
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Muse 2 Signal Diagnostic v2")
        self.window.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

        self.layout = QtWidgets.QVBoxLayout(self.window)

        self.status_label = QtWidgets.QLabel("Starting Muse 2 diagnostic viewer...")
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.layout.addWidget(self.status_label)

        self.help_label = QtWidgets.QLabel(self._help_text())
        self.help_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.layout.addWidget(self.help_label)

        self.graphics = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.graphics)

        self._setup_plots()
        self._setup_shortcuts()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        self.app.aboutToQuit.connect(self.cleanup)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def _get_channel_map(self):
        eeg_channels = self.descr["eeg_channels"]
        eeg_names = self.descr["eeg_names"].split(",")

        channel_map = dict(zip(eeg_names, eeg_channels))

        missing = [name for name in CHANNEL_NAMES if name not in channel_map]
        if missing:
            raise RuntimeError(f"Missing expected Muse 2 channels: {missing}")

        return channel_map

    def _setup_session_files(self):
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.session_dir = Path(self.args.output_root) / session_name
        self.session_dir.mkdir(parents=True, exist_ok=False)

        labels_path = self.session_dir / "labels.json"
        config_path = self.session_dir / "session_config.json"
        readme_path = self.session_dir / "README_session.txt"

        with labels_path.open("w", encoding="utf-8") as f:
            json.dump(LABELS, f, indent=2)

        config = {
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "created_at_local": datetime.now().isoformat(timespec="seconds"),
            "board_id": BOARD_ID,
            "board_name": "Muse 2",
            "sampling_rate_hz": self.sampling_rate,
            "channel_map": self.channel_map,
            "raw_columns": RAW_COLUMNS,
            "event_columns": EVENT_COLUMNS,
            "labels": LABELS,
            "window_seconds": self.args.window_seconds,
            "feature_window_seconds": self.args.feature_window_seconds,
            "feature_history_seconds": self.args.feature_history_seconds,
            "frontal_threshold_uv": self.args.frontal_threshold_uv,
            "diff_threshold_uv": self.args.diff_threshold_uv,
            "temporal_threshold_uv": self.args.temporal_threshold_uv,
            "fast_change_threshold_uv": self.args.fast_change_threshold_uv,
            "suspicious_extreme_uv": self.args.suspicious_extreme_uv,
            "notes": (
                "Manual labeled Muse 2 EEG session. Data is local biometric data "
                "and should not be pushed to Git."
            ),
        }

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        readme_text = (
            "Muse 2 manual event labeling session\n"
            "\n"
            "Files:\n"
            "- raw_eeg.tsv: continuous EEG stream and BrainFlow metadata rows.\n"
            "- events.tsv: manually labeled event segments.\n"
            "- labels.json: key-to-label mapping used during this session.\n"
            "- session_config.json: acquisition and script configuration.\n"
            "\n"
            "Keyboard controls:\n"
            "0-9 = select label\n"
            "R = start/stop segment\n"
            "U = undo last completed segment\n"
            "ESC = cancel current open segment\n"
            "Q = quit cleanly\n"
            "\n"
            "This folder contains local biometric data and should not be committed.\n"
        )

        with readme_path.open("w", encoding="utf-8") as f:
            f.write(readme_text)

        raw_path = self.session_dir / "raw_eeg.tsv"
        self.raw_file_handle = raw_path.open("w", newline="", encoding="utf-8")
        self.raw_writer = csv.DictWriter(
            self.raw_file_handle,
            fieldnames=RAW_COLUMNS,
            delimiter="\t",
        )
        self.raw_writer.writeheader()
        self.raw_file_handle.flush()

        self._write_events_file()

        print(f"Save mode enabled.")
        print(f"Session folder: {self.session_dir.resolve()}")
        print()

    def _setup_plots(self):
        self.raw_plot = self.graphics.addPlot(row=0, col=0)
        self.raw_plot.setTitle("Centered EEG channels with vertical offsets")
        self.raw_plot.setLabel("left", "uV + offset")
        self.raw_plot.setLabel("bottom", "seconds")
        self.raw_plot.addLegend()
        self.raw_plot.setXRange(-self.args.window_seconds, 0, padding=0)

        self.raw_offsets = {
            "TP9": 900.0,
            "AF7": 300.0,
            "AF8": -300.0,
            "TP10": -900.0,
        }

        self.raw_curves = {}
        for name in CHANNEL_NAMES:
            self.raw_curves[name] = self.raw_plot.plot(name=f"{name} centered")

        self.derived_plot = self.graphics.addPlot(row=1, col=0)
        self.derived_plot.setTitle("Derived diagnostic traces")
        self.derived_plot.setLabel("left", "uV")
        self.derived_plot.setLabel("bottom", "seconds")
        self.derived_plot.addLegend()
        self.derived_plot.setXRange(-self.args.window_seconds, 0, padding=0)

        self.frontal_common_curve = self.derived_plot.plot(name="(AF7 + AF8) / 2")
        self.frontal_diff_curve = self.derived_plot.plot(name="(AF7 - AF8) / 2")
        self.temporal_common_curve = self.derived_plot.plot(name="(TP9 + TP10) / 2")

        self.feature_plot = self.graphics.addPlot(row=2, col=0)
        self.feature_plot.setTitle("Recent feature history")
        self.feature_plot.setLabel("left", "peak-to-peak uV")
        self.feature_plot.setLabel("bottom", "seconds")
        self.feature_plot.addLegend()
        self.feature_plot.setXRange(-self.args.feature_history_seconds, 0, padding=0)

        self.frontal_feature_curve = self.feature_plot.plot(name="frontal common p2p")
        self.diff_feature_curve = self.feature_plot.plot(name="AF7-AF8 diff p2p")
        self.temporal_feature_curve = self.feature_plot.plot(name="temporal common p2p")

        self.feature_plot.addItem(
            pg.InfiniteLine(
                pos=self.args.frontal_threshold_uv,
                angle=0,
                movable=False,
                label="front threshold",
            )
        )
        self.feature_plot.addItem(
            pg.InfiniteLine(
                pos=self.args.diff_threshold_uv,
                angle=0,
                movable=False,
                label="diff threshold",
            )
        )
        self.feature_plot.addItem(
            pg.InfiniteLine(
                pos=self.args.temporal_threshold_uv,
                angle=0,
                movable=False,
                label="temporal threshold",
            )
        )

    def _setup_shortcuts(self):
        self.shortcuts = []

        def add_shortcut(sequence, callback):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(sequence), self.window)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

        for key in LABELS:
            add_shortcut(key, lambda key=key: self.select_label(key))

        add_shortcut("R", self.toggle_segment)
        add_shortcut("U", self.undo_last_segment)
        add_shortcut("Esc", self.cancel_current_segment)
        add_shortcut("Q", self.quit_cleanly)

    def _help_text(self):
        if self.args.save:
            return (
                "SAVE MODE | 0-9 select label | R start/stop recording | "
                "U undo last | ESC cancel current | Q quit"
            )

        return "DIAGNOSTIC MODE | Q quit | No data is saved."

    def start(self):
        print("Starting Muse 2 Signal Diagnostic v2")
        print("Close the Muse mobile app before connecting.")
        print("Turn off phone Bluetooth if it keeps taking the Muse connection.")
        print("This script does not control the motor.")
        print()

        if self.args.save:
            print("This run WILL save local biometric data.")
        else:
            print("This run will NOT save data.")

        print()
        print(f"Board: Muse 2")
        print(f"Sampling rate: {self.sampling_rate} Hz")
        print(f"Channels: {self.channel_map}")
        print()

        self.board.prepare_session()
        self.board.start_stream()

        self.window.resize(1250, 950)
        self.window.show()
        self.window.activateWindow()
        self.window.setFocus()

        self.timer.start(self.args.update_ms)

        sys.exit(self.app.exec())

    def update(self):
        new_data = self.board.get_board_data()

        if new_data.shape[1] > 0:
            self._ingest_new_data(new_data)

        if len(self.display_times) < self.feature_samples:
            self.status_label.setText(
                f"Waiting for data... samples={len(self.display_times)} / {self.feature_samples}"
            )
            return

        self._update_plots_and_status()

    def _ingest_new_data(self, data):
        n_samples = data.shape[1]

        for i in range(n_samples):
            sample_index = self.total_samples
            time_s = sample_index / self.sampling_rate

            values = {}
            for name in CHANNEL_NAMES:
                values[name] = float(data[self.channel_map[name], i])

            package_num = self._read_optional_channel(data, self.package_num_channel, i)
            timestamp = self._read_optional_channel(data, self.timestamp_channel, i)
            marker = self._read_optional_channel(data, self.marker_channel, i)
            other = self._read_other_channel(data, i)

            row = {
                "sample_index": sample_index,
                "time_s": time_s,
                "package_num": package_num,
                "TP9": values["TP9"],
                "AF7": values["AF7"],
                "AF8": values["AF8"],
                "TP10": values["TP10"],
                "other": other,
                "timestamp": timestamp,
                "marker": marker,
            }

            self.display_times.append(time_s)
            for name in CHANNEL_NAMES:
                self.display_buffers[name].append(values[name])

            if self.args.save:
                self.raw_rows.append(row)
                self.raw_writer.writerow(row)

            self.total_samples += 1

        if self.args.save and self.raw_file_handle is not None:
            self.raw_file_handle.flush()

    def _read_optional_channel(self, data, channel_index, sample_col):
        if channel_index is None:
            return np.nan

        try:
            return float(data[channel_index, sample_col])
        except Exception:
            return np.nan

    def _read_other_channel(self, data, sample_col):
        if not self.other_channels:
            return np.nan

        try:
            return float(data[self.other_channels[0], sample_col])
        except Exception:
            return np.nan

    def _update_plots_and_status(self):
        times = np.asarray(self.display_times, dtype=float)
        x = times - times[-1]

        raw = {}
        centered = {}

        for name in CHANNEL_NAMES:
            raw[name] = np.asarray(self.display_buffers[name], dtype=float)
            centered[name] = raw[name] - np.median(raw[name])

        for name in CHANNEL_NAMES:
            self.raw_curves[name].setData(x, centered[name] + self.raw_offsets[name])

        af7 = centered["AF7"]
        af8 = centered["AF8"]
        tp9 = centered["TP9"]
        tp10 = centered["TP10"]

        frontal_common = (af7 + af8) / 2.0
        frontal_diff = (af7 - af8) / 2.0
        temporal_common = (tp9 + tp10) / 2.0

        self.frontal_common_curve.setData(x, frontal_common)
        self.frontal_diff_curve.setData(x, frontal_diff)
        self.temporal_common_curve.setData(x, temporal_common)

        frontal_seg = frontal_common[-self.feature_samples :]
        diff_seg = frontal_diff[-self.feature_samples :]
        temporal_seg = temporal_common[-self.feature_samples :]
        af7_seg = af7[-self.feature_samples :]
        af8_seg = af8[-self.feature_samples :]

        features = self._compute_features(
            frontal_seg,
            diff_seg,
            temporal_seg,
            af7_seg,
            af8_seg,
            raw,
        )

        elapsed = times[-1]
        self._append_feature_history(
            elapsed,
            features["frontal_common_p2p_uv"],
            features["frontal_diff_p2p_uv"],
            features["temporal_common_p2p_uv"],
        )
        self._update_feature_plot(elapsed)

        self._maybe_print_diagnostic_event(features)
        self._update_status_label(features)

    def _compute_features(self, frontal_seg, diff_seg, temporal_seg, af7_seg, af8_seg, raw):
        recent_raw_values = []
        for name in CHANNEL_NAMES:
            recent_raw_values.append(raw[name][-self.feature_samples :])

        recent_raw_values = np.concatenate(recent_raw_values)

        max_abs_raw = float(np.max(np.abs(recent_raw_values)))
        suspicious_extreme = bool(max_abs_raw >= self.args.suspicious_extreme_uv)

        return {
            "frontal_common_p2p_uv": peak_to_peak(frontal_seg),
            "frontal_diff_p2p_uv": peak_to_peak(diff_seg),
            "temporal_common_p2p_uv": peak_to_peak(temporal_seg),
            "frontal_step_uv_per_sample": max_abs_step(frontal_seg),
            "af7_af8_corr": safe_corr(af7_seg, af8_seg),
            "max_abs_raw_uv": max_abs_raw,
            "suspicious_extreme": suspicious_extreme,
        }

    def _append_feature_history(
        self,
        elapsed,
        frontal_common_p2p,
        frontal_diff_p2p,
        temporal_common_p2p,
    ):
        self.feature_times.append(elapsed)
        self.frontal_history.append(frontal_common_p2p)
        self.diff_history.append(frontal_diff_p2p)
        self.temporal_history.append(temporal_common_p2p)

        min_time = elapsed - self.args.feature_history_seconds

        while self.feature_times and self.feature_times[0] < min_time:
            self.feature_times.popleft()
            self.frontal_history.popleft()
            self.diff_history.popleft()
            self.temporal_history.popleft()

    def _update_feature_plot(self, elapsed):
        if not self.feature_times:
            return

        t = np.asarray(self.feature_times, dtype=float)
        x = t - elapsed

        self.frontal_feature_curve.setData(x, np.asarray(self.frontal_history))
        self.diff_feature_curve.setData(x, np.asarray(self.diff_history))
        self.temporal_feature_curve.setData(x, np.asarray(self.temporal_history))

    def _maybe_print_diagnostic_event(self, features):
        now = time.monotonic()

        if now - self.last_console_event_time < self.args.cooldown_seconds:
            return

        candidate = candidate_label_from_features(
            features["frontal_common_p2p_uv"],
            features["frontal_diff_p2p_uv"],
            features["temporal_common_p2p_uv"],
            features["frontal_step_uv_per_sample"],
            features["af7_af8_corr"],
            self.args,
        )

        if candidate == "":
            return

        self.last_console_event_time = now

        print(
            "[EVENT] "
            f"{candidate} | "
            f"front_p2p={features['frontal_common_p2p_uv']:.1f} uV | "
            f"diff_p2p={features['frontal_diff_p2p_uv']:.1f} uV | "
            f"temporal_p2p={features['temporal_common_p2p_uv']:.1f} uV | "
            f"front_step={features['frontal_step_uv_per_sample']:.1f} uV/sample | "
            f"AF7_AF8_corr={features['af7_af8_corr']:.2f} | "
            f"suspicious_extreme={features['suspicious_extreme']}"
        )

    def _update_status_label(self, features):
        mode = "SAVE" if self.args.save else "DIAGNOSTIC"

        if self.recording_segment:
            state = (
                f"RECORDING {self.selected_label} | "
                f"elapsed={self.current_segment_elapsed_s():.2f} s"
            )
        else:
            state = "IDLE"

        counts = self._valid_counts_by_label()
        selected_count = counts.get(self.selected_label, 0)
        total_valid = sum(counts.values())

        session_text = ""
        if self.args.save and self.session_dir is not None:
            session_text = f" | session={self.session_dir.name}"

        status = (
            f"Mode: {mode}{session_text}\n"
            f"Selected: {self.selected_label_key} = {self.selected_label} | "
            f"State: {state} | "
            f"{self.selected_label} recorded: {selected_count} | "
            f"Total valid trials: {total_valid}\n"
            f"Last saved: {self.last_saved_message}\n"
            f"front_p2p={features['frontal_common_p2p_uv']:.1f} uV | "
            f"diff_p2p={features['frontal_diff_p2p_uv']:.1f} uV | "
            f"temporal_p2p={features['temporal_common_p2p_uv']:.1f} uV | "
            f"corr={features['af7_af8_corr']:.2f} | "
            f"extreme={features['suspicious_extreme']}"
        )

        self.status_label.setText(status)

    def select_label(self, label_key):
        if label_key not in LABELS:
            return

        if self.recording_segment:
            print("Cannot change label while recording. Press ESC to cancel or R to finish.")
            return

        self.selected_label_key = label_key
        self.selected_label = LABELS[label_key]
        self.last_saved_message = f"Selected {label_key} = {self.selected_label}"
        print(f"[LABEL] {label_key} = {self.selected_label}")

    def toggle_segment(self):
        if not self.args.save:
            print("R ignored: run with --save to label segments.")
            return

        if self.recording_segment:
            self.stop_segment()
        else:
            self.start_segment()

    def start_segment(self):
        if self.total_samples == 0:
            print("Cannot start segment yet: no samples received.")
            return

        self.recording_segment = True
        self.segment_start_sample = self.total_samples
        self.segment_start_time_s = self.segment_start_sample / self.sampling_rate
        self.last_saved_message = f"Recording {self.selected_label}..."
        print(
            f"[START] {self.selected_label} | "
            f"start_sample={self.segment_start_sample} | "
            f"start_time={self.segment_start_time_s:.3f}s"
        )

    def stop_segment(self):
        if not self.recording_segment:
            return

        end_sample = self.total_samples - 1
        end_time_s = end_sample / self.sampling_rate

        if end_sample < self.segment_start_sample:
            print("Segment too short or no samples captured. Canceled.")
            self.cancel_current_segment()
            return

        event = self._build_event(
            self.selected_label_key,
            self.selected_label,
            self.segment_start_sample,
            end_sample,
            self.segment_start_time_s,
            end_time_s,
        )

        self.events.append(event)
        self.recording_segment = False
        self.segment_start_sample = None
        self.segment_start_time_s = None

        self._write_events_file()
        self._write_summary_file()

        duration_s = float(event["duration_s"])

        self.last_saved_message = (
            f"{event['label']} #{event['label_trial_index']} | "
            f"{duration_s:.3f}s"
        )

        print(
            f"[SAVED] {event['label']} #{event['label_trial_index']} | "
            f"samples={event['n_samples']} | "
            f"duration={duration_s:.3f}s"
        )

    def _build_event(
        self,
        label_key,
        label,
        start_sample,
        end_sample,
        start_time_s,
        end_time_s,
    ):
        self.event_counter += 1

        label_trial_index = 1
        for event in self.events:
            if event["label"] == label and event["valid"] == "true":
                label_trial_index += 1

        segment_rows = self.raw_rows[start_sample : end_sample + 1]
        features = self._features_for_segment(segment_rows)

        duration_s = end_time_s - start_time_s
        n_samples = end_sample - start_sample + 1

        event = {
            "event_id": self.event_counter,
            "label_key": label_key,
            "label": label,
            "label_trial_index": label_trial_index,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_time_s": f"{start_time_s:.6f}",
            "end_time_s": f"{end_time_s:.6f}",
            "duration_s": f"{duration_s:.6f}",
            "n_samples": n_samples,
            "valid": "true",
            "rejected_reason": "",
            **features,
        }

        return event

    def _features_for_segment(self, segment_rows):
        if not segment_rows:
            return {
                "frontal_common_p2p_uv": np.nan,
                "frontal_diff_p2p_uv": np.nan,
                "temporal_common_p2p_uv": np.nan,
                "frontal_step_uv_per_sample": np.nan,
                "af7_af8_corr": np.nan,
                "max_abs_raw_uv": np.nan,
                "suspicious_extreme": "unknown",
            }

        tp9 = np.asarray([row["TP9"] for row in segment_rows], dtype=float)
        af7 = np.asarray([row["AF7"] for row in segment_rows], dtype=float)
        af8 = np.asarray([row["AF8"] for row in segment_rows], dtype=float)
        tp10 = np.asarray([row["TP10"] for row in segment_rows], dtype=float)

        tp9_c = tp9 - np.median(tp9)
        af7_c = af7 - np.median(af7)
        af8_c = af8 - np.median(af8)
        tp10_c = tp10 - np.median(tp10)

        frontal_common = (af7_c + af8_c) / 2.0
        frontal_diff = (af7_c - af8_c) / 2.0
        temporal_common = (tp9_c + tp10_c) / 2.0

        raw_values = np.concatenate([tp9, af7, af8, tp10])
        max_abs_raw = float(np.max(np.abs(raw_values)))
        suspicious_extreme = bool(max_abs_raw >= self.args.suspicious_extreme_uv)

        return {
            "frontal_common_p2p_uv": f"{peak_to_peak(frontal_common):.6f}",
            "frontal_diff_p2p_uv": f"{peak_to_peak(frontal_diff):.6f}",
            "temporal_common_p2p_uv": f"{peak_to_peak(temporal_common):.6f}",
            "frontal_step_uv_per_sample": f"{max_abs_step(frontal_common):.6f}",
            "af7_af8_corr": f"{safe_corr(af7_c, af8_c):.6f}",
            "max_abs_raw_uv": f"{max_abs_raw:.6f}",
            "suspicious_extreme": str(suspicious_extreme).lower(),
        }

    def undo_last_segment(self):
        if not self.args.save:
            print("Undo ignored: run with --save to label segments.")
            return

        if self.recording_segment:
            print("Undo ignored while recording. Press ESC to cancel current segment.")
            return

        for event in reversed(self.events):
            if event["valid"] == "true":
                event["valid"] = "false"
                event["rejected_reason"] = "manual_undo"
                self.last_saved_message = f"Undo: {event['label']} event_id={event['event_id']}"
                self._write_events_file()
                self._write_summary_file()
                print(f"[UNDO] event_id={event['event_id']} label={event['label']}")
                return

        print("No valid segment to undo.")

    def cancel_current_segment(self):
        if not self.args.save:
            return

        if not self.recording_segment:
            print("No open segment to cancel.")
            return

        print(f"[CANCEL] {self.selected_label}")
        self.recording_segment = False
        self.segment_start_sample = None
        self.segment_start_time_s = None
        self.last_saved_message = f"Canceled {self.selected_label}"

    def quit_cleanly(self):
        if self.recording_segment:
            print("[QUIT] Open segment canceled before exit.")
            self.cancel_current_segment()

        self._write_summary_file()
        self.app.quit()

    def current_segment_elapsed_s(self):
        if not self.recording_segment or self.segment_start_sample is None:
            return 0.0

        current_time_s = self.total_samples / self.sampling_rate
        return current_time_s - self.segment_start_time_s

    def _valid_counts_by_label(self):
        counts = {}

        for event in self.events:
            if event["valid"] != "true":
                continue

            label = event["label"]
            counts[label] = counts.get(label, 0) + 1

        return counts

    def _write_events_file(self):
        if not self.args.save or self.session_dir is None:
            return

        events_path = self.session_dir / "events.tsv"
        temp_path = self.session_dir / "events.tsv.tmp"

        with temp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=EVENT_COLUMNS,
                delimiter="\t",
            )
            writer.writeheader()
            for event in self.events:
                writer.writerow(event)

        temp_path.replace(events_path)

    def _write_summary_file(self):
        if not self.args.save or self.session_dir is None:
            return

        summary_path = self.session_dir / "session_summary.json"
        counts = self._valid_counts_by_label()

        summary = {
            "script_name": SCRIPT_NAME,
            "script_version": SCRIPT_VERSION,
            "updated_at_local": datetime.now().isoformat(timespec="seconds"),
            "session_dir": str(self.session_dir),
            "total_samples": self.total_samples,
            "duration_s": self.total_samples / self.sampling_rate,
            "total_events": len(self.events),
            "total_valid_events": sum(counts.values()),
            "valid_counts_by_label": counts,
            "labels": LABELS,
        }

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True

        print()
        print("Cleaning up BrainFlow session...")

        try:
            if self.args.save:
                self._write_events_file()
                self._write_summary_file()

            if self.raw_file_handle is not None:
                self.raw_file_handle.flush()
                self.raw_file_handle.close()

            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()

        except Exception as exc:
            print(f"Cleanup warning: {exc}")

        print("Done.")

        if self.args.save and self.session_dir is not None:
            print(f"Session saved at: {self.session_dir.resolve()}")

    def handle_sigint(self, *_):
        self.quit_cleanly()


def main():
    args = parse_args()
    viewer = MuseSignalDiagnosticViewer(args)
    viewer.start()


if __name__ == "__main__":
    main()