import sys
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


DISPLAY_SECONDS = 5.0
FEATURE_WINDOW_SECONDS = 0.25
UPDATE_MS = 40


def main():
    params = BrainFlowInputParams()
    params.other_info = "p21"

    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)

    descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(descr["sampling_rate"])
    eeg_channels = descr["eeg_channels"]
    eeg_names = descr["eeg_names"].split(",")

    channel_by_name = dict(zip(eeg_names, eeg_channels))

    af7_channel = channel_by_name["AF7"]
    af8_channel = channel_by_name["AF8"]

    display_samples = int(DISPLAY_SECONDS * sampling_rate)
    feature_samples = int(FEATURE_WINDOW_SECONDS * sampling_rate)

    print("Preparing Muse 2 session...")
    board.prepare_session()
    board.start_stream()

    print("Muse 2 stream started.")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"AF7 channel index: {af7_channel}")
    print(f"AF8 channel index: {af8_channel}")
    print("Close the plot window to stop.")

    app = QtWidgets.QApplication(sys.argv)

    win = pg.GraphicsLayoutWidget(title="Muse 2 live blink viewer")
    win.resize(1000, 700)

    signal_plot = win.addPlot(title="AF7 and AF8 live EEG, median-centered")
    signal_plot.setLabel("left", "Amplitude", units="uV")
    signal_plot.setLabel("bottom", "Time", units="s")
    signal_plot.showGrid(x=True, y=True)

    af7_curve = signal_plot.plot(name="AF7")
    af8_curve = signal_plot.plot(name="AF8")

    win.nextRow()

    feature_plot = win.addPlot(title="Recent blink feature")
    feature_plot.setLabel("left", "Peak-to-peak", units="uV")
    feature_plot.setLabel("bottom", "Time", units="s")
    feature_plot.showGrid(x=True, y=True)

    feature_curve = feature_plot.plot()

    feature_times = []
    feature_values = []

    start_time = time.time()

    def update():
        data = board.get_current_board_data(display_samples)

        if data.shape[1] < 10:
            return

        n_samples = data.shape[1]
        x = np.arange(n_samples) / sampling_rate
        x = x - x[-1]

        af7 = data[af7_channel, :]
        af8 = data[af8_channel, :]

        af7_centered = af7 - np.median(af7)
        af8_centered = af8 - np.median(af8)

        af7_curve.setData(x, af7_centered)
        af8_curve.setData(x, af8_centered)

        recent_af7 = af7_centered[-feature_samples:]
        recent_af8 = af8_centered[-feature_samples:]

        af7_p2p = float(np.max(recent_af7) - np.min(recent_af7))
        af8_p2p = float(np.max(recent_af8) - np.min(recent_af8))
        feature = max(af7_p2p, af8_p2p)

        now = time.time() - start_time
        feature_times.append(now)
        feature_values.append(feature)

        while feature_times and now - feature_times[0] > 20:
            feature_times.pop(0)
            feature_values.pop(0)

        feature_curve.setData(feature_times, feature_values)

        signal_plot.setTitle(
            f"AF7/AF8 live EEG | recent p2p feature: {feature:.1f} uV"
        )

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(UPDATE_MS)

    win.show()

    try:
        app.exec()
    finally:
        print("Stopping Muse 2 stream...")
        try:
            board.stop_stream()
        except Exception:
            pass
        board.release_session()
        print("Muse session released.")


if __name__ == "__main__":
    main()