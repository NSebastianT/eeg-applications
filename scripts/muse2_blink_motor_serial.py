import time
from collections import deque

import numpy as np
import serial

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


SERIAL_PORT = "COM7"
BAUD_RATE = 115200

PWM_VALUE = 255

REQUIRED_EVENTS = 1

TOGGLE_COOLDOWN_SECONDS = 1.0

DETECTION_THRESHOLD_UV = 900.0
RESET_THRESHOLD_UV = 600.0

GESTURE_HOLD_SECONDS = 0.40
WINDOW_SECONDS = 0.25
EVENT_WINDOW_SECONDS = 1.5
HEARTBEAT_SECONDS = 0.35


def blink_feature(data_window, af7_channel, af8_channel):
    if data_window.shape[1] == 0:
        return None

    af7 = data_window[af7_channel, :]
    af8 = data_window[af8_channel, :]

    af7 = af7 - np.median(af7)
    af8 = af8 - np.median(af8)

    af7_p2p = float(np.max(af7) - np.min(af7))
    af8_p2p = float(np.max(af8) - np.min(af8))

    return max(af7_p2p, af8_p2p)


def send_forward(serial_port):
    serial_port.write(f"FWD {PWM_VALUE}\n".encode("utf-8"))


def send_stop(serial_port):
    serial_port.write(b"STOP\n")


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

    window_samples = int(WINDOW_SECONDS * sampling_rate)

    motor_on = False
    signal_armed = True
    event_times = deque()

    gesture_start_time = None
    last_toggle_time = -999.0
    last_heartbeat_time = 0.0

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)

    try:
        print("Preparing Muse 2 session...")
        board.prepare_session()
        board.start_stream()

        print("Muse 2 stream started.")
        print(f"Sampling rate: {sampling_rate} Hz")
        print(f"AF7 channel index: {af7_channel}")
        print(f"AF8 channel index: {af8_channel}")
        print()
        print(f"Required events: {REQUIRED_EVENTS}")
        print(f"Detection threshold: {DETECTION_THRESHOLD_UV:.1f} uV")
        print(f"Reset threshold: {RESET_THRESHOLD_UV:.1f} uV")
        print(f"Gesture hold time: {GESTURE_HOLD_SECONDS:.2f} s")
        print(f"Toggle cooldown: {TOGGLE_COOLDOWN_SECONDS:.1f} s")
        print()
        print("Hold a clear blink / eye squeeze / eyebrow gesture for about 0.4 s.")
        print("Event count reached = toggle motor ON/OFF.")
        print("Press Ctrl+C to stop.")
        print()

        send_stop(ser)

        while True:
            now = time.time()
            data = board.get_current_board_data(window_samples)

            if data.shape[1] >= window_samples:
                feature = blink_feature(data, af7_channel, af8_channel)

                if feature is not None:
                    in_cooldown = (now - last_toggle_time) < TOGGLE_COOLDOWN_SECONDS

                    if feature >= DETECTION_THRESHOLD_UV:
                        if gesture_start_time is None:
                            gesture_start_time = now

                        gesture_duration = now - gesture_start_time

                        if (
                            signal_armed
                            and not in_cooldown
                            and gesture_duration >= GESTURE_HOLD_SECONDS
                        ):
                            signal_armed = False
                            event_times.append(now)

                            while event_times and now - event_times[0] > EVENT_WINDOW_SECONDS:
                                event_times.popleft()

                            print(
                                f"Gesture event | feature={feature:.1f} uV | "
                                f"held={gesture_duration:.2f} s | "
                                f"count={len(event_times)}/{REQUIRED_EVENTS}"
                            )

                            if len(event_times) >= REQUIRED_EVENTS:
                                motor_on = not motor_on
                                last_toggle_time = now
                                event_times.clear()

                                if motor_on:
                                    print("MOTOR ON")
                                    send_forward(ser)
                                    last_heartbeat_time = now
                                else:
                                    print("MOTOR OFF")
                                    send_stop(ser)

                    else:
                        gesture_start_time = None

                    if not signal_armed and feature <= RESET_THRESHOLD_UV:
                        signal_armed = True
                        gesture_start_time = None
                        print(f"Signal reset | feature={feature:.1f} uV")

            if motor_on and (time.time() - last_heartbeat_time > HEARTBEAT_SECONDS):
                send_forward(ser)
                last_heartbeat_time = time.time()

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        send_stop(ser)
        ser.close()

        try:
            board.stop_stream()
        except Exception:
            pass

        board.release_session()
        print("Motor stopped. Muse session released.")


if __name__ == "__main__":
    main()