from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BoardIds


INPUT_FILE = Path("data/processed/muse2_labeled.csv")
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


def main():
    board_id = BoardIds.MUSE_2_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    df = pd.read_csv(INPUT_FILE)

    print("File:", INPUT_FILE)
    print("Shape:", df.shape)
    print("Sampling rate:", sampling_rate, "Hz")

    if "time_s" in df.columns:
        print("Duration by sample count:", df["time_s"].iloc[-1])

    print()

    for ch in EEG_CHANNELS:
        near_low = (df[ch] <= -990).sum()
        near_high = (df[ch] >= 990).sum()
        total = len(df)

        print(ch)
        print("  <= -990:", near_low, f"({near_low / total * 100:.2f}%)")
        print("  >=  990:", near_high, f"({near_high / total * 100:.2f}%)")

    first_10s = df[df["time_s"] <= 10]

    plt.figure()
    for ch in EEG_CHANNELS:
        plt.plot(first_10s["time_s"], first_10s[ch], label=ch)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Muse 2 raw EEG - first 10 seconds")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()