\# eeg-applications



Using the Muse 2 device for different engineering applications.



\## Current status



Initial Muse 2 live EEG acquisition was tested using Python and BrainFlow.



The current pipeline records raw EEG data from the Muse 2 headband over Bluetooth Low Energy and saves it as a local CSV file.



Muse 2 provides four EEG channels exposed through BrainFlow:



\- TP9

\- AF7

\- AF8

\- TP10



In the current Python/BrainFlow setup, the reported sampling rate is 256 Hz.



The device includes multiple physical contact/sensor points, while the EEG data stream used in this project exposes four EEG channels.



\## Local pipeline



```text

Muse 2

→ Bluetooth Low Energy

→ Python / BrainFlow

→ Raw CSV

→ Labeled CSV

→ Basic signal quality check

