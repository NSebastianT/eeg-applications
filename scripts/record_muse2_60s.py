import time
from pathlib import Path

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter


OUTPUT_FILE = Path("data/raw/muse2_raw_60s.csv")
RECORD_SECONDS = 60


def main():
    BoardShim.enable_board_logger()

    params = BrainFlowInputParams()
    board_id = BoardIds.MUSE_2_BOARD.value
    board = BoardShim(board_id, params)

    try:
        print("Preparing Muse 2 session...")
        board.prepare_session()

        print(f"Recording {RECORD_SECONDS} seconds...")
        board.start_stream()
        time.sleep(RECORD_SECONDS)

        data = board.get_board_data()
        board.stop_stream()

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        DataFilter.write_file(data, str(OUTPUT_FILE), "w")

        eeg_channels = BoardShim.get_eeg_channels(board_id)
        eeg_names = BoardShim.get_eeg_names(board_id)
        sampling_rate = BoardShim.get_sampling_rate(board_id)

        print("Done.")
        print(f"Saved file: {OUTPUT_FILE}")
        print(f"Data shape: {data.shape}")
        print(f"EEG channel indexes: {eeg_channels}")
        print(f"EEG channel names: {eeg_names}")
        print(f"Sampling rate: {sampling_rate} Hz")

    finally:
        if board.is_prepared():
            board.release_session()


if __name__ == "__main__":
    main()