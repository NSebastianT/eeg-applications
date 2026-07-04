from pathlib import Path

import pandas as pd
from brainflow.board_shim import BoardShim, BoardIds


INPUT_FILE = Path("data/raw/muse2_raw_60s.csv")
OUTPUT_FILE = Path("data/processed/muse2_labeled.csv")


def main():
    board_id = BoardIds.MUSE_2_BOARD.value
    sampling_rate = BoardShim.get_sampling_rate(board_id)

    df = pd.read_csv(INPUT_FILE, sep="\t", header=None)

    df.columns = [
        "package_num",
        "TP9",
        "AF7",
        "AF8",
        "TP10",
        "other",
        "timestamp",
        "marker",
    ]

    df["sample_index"] = range(len(df))
    df["time_s"] = df["sample_index"] / sampling_rate

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()