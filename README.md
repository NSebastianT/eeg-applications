# eeg-applications

Using the Muse 2 device for different engineering applications.

## Current status

Initial Muse 2 live EEG acquisition was tested using Python and BrainFlow.

The current pipeline records raw EEG data from the Muse 2 headband over Bluetooth Low Energy and saves it as a local CSV file.

Muse 2 provides four EEG channels exposed through BrainFlow:

- TP9
- AF7
- AF8
- TP10

In the current Python/BrainFlow setup, the reported sampling rate is 256 Hz.

The device includes multiple physical contact/sensor points, while the EEG data stream used in this project exposes four EEG channels.

## Local pipeline

- Muse 2
- Bluetooth Low Energy
- Python / BrainFlow
- Raw CSV
- Labeled CSV
- Basic signal quality check

## Scripts

### `scripts/record_muse2_60s.py`

Records a 60-second raw EEG session from Muse 2 and saves it locally.

Default output:

`data/raw/muse2_raw_60s.csv`

### `scripts/label_muse2_csv.py`

Loads the raw BrainFlow CSV, assigns channel names, adds sample index and time in seconds, and saves a labeled CSV.

Default input:

`data/raw/muse2_raw_60s.csv`

Default output:

`data/processed/muse2_labeled.csv`

### `scripts/check_signal_quality.py`

Checks basic saturation near ±990 in the four EEG channels and plots the first 10 seconds of raw EEG.

Default input:

`data/processed/muse2_labeled.csv`

## Repository data policy

Raw EEG data files are not committed to the repository.

The repository keeps the data folder structure with `.gitkeep` files:

- `data/raw/.gitkeep`
- `data/processed/.gitkeep`

CSV files are ignored by Git to avoid committing local biometric data by accident.

## Environment

The current local environment uses:

- Python 3.11
- BrainFlow
- NumPy
- pandas
- Matplotlib
- SciPy
- MNE
- pyqtgraph
- setuptools 80.10.2

`setuptools==80.10.2` is pinned because the current BrainFlow setup in this environment requires `pkg_resources` during DLL loading.

## Current verified baseline

A second 60-second baseline recording was generated locally and used to validate the pipeline.

Local file:

`muse2_raw_60s_baseline_02.csv`

This file is not committed to the repository.

The recording produced:

- Raw BrainFlow shape: `(8, 15336)`
- Labeled CSV shape: `(15336, 10)`
- Sampling rate: `256 Hz`
- Duration by sample count: approximately `59.90 s`
- Saturation near ±990: `0.00%` in TP9, AF7, AF8, and TP10

## Notes

This repository currently contains the initial acquisition pipeline only.

Filtering, frequency-band analysis, machine learning, ROS2, Raspberry Pi integration, and hardware control have not been implemented yet.