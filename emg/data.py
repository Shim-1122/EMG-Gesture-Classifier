"""Download and load the UCI "EMG Data for Gestures" dataset.

36 subjects wore a Myo armband (8 surface EMG channels around the forearm) and
performed 6 gestures in two separate sessions. The raw files have irregular
timestamps, so every recording is resampled onto a uniform time grid before any
filtering (digital filters assume evenly spaced samples).
"""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FS = 200  # Hz, uniform rate after resampling (the Myo armband's native EMG rate)
N_CHANNELS = 8

GESTURES = {
    1: "Rest",
    2: "Fist",
    3: "Wrist flexion",
    4: "Wrist extension",
    5: "Radial deviation",
    6: "Ulnar deviation",
}


def download(data_dir: Path = DATA_DIR) -> Path:
    """Download and unzip the dataset on first use. Returns the root folder (one subfolder per subject)."""
    root = Path(data_dir) / "EMG_data_for_gestures-master"
    if not root.exists():
        print(f"Downloading dataset (~17 MB) to {data_dir} ...")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(data_dir)
    return root


def load_recording(path, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    """Load one session file and resample it to `fs`.

    Returns
    -------
    emg    : (n_samples, 8) float array in volts
    labels : (n_samples,) int array, gesture id per sample (0 = unlabeled)
    """
    raw = pd.read_csv(path, sep="\t").to_numpy()
    t = raw[:, 0] / 1000.0                       # file stores milliseconds
    t_uniform = np.arange(t[0], t[-1], 1 / fs)

    emg = np.column_stack([np.interp(t_uniform, t, raw[:, ch]) for ch in range(1, N_CHANNELS + 1)])

    # Labels are categories, so never interpolate them: take the most recent raw label (zero-order hold)
    last_raw_index = np.searchsorted(t, t_uniform, side="right") - 1
    labels = raw[last_raw_index, -1].astype(int)
    return emg, labels


def load_all(root=None) -> dict[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Load every subject. Returns {subject_id: [session_1, session_2]}, each session an (emg, labels) pair."""
    root = Path(root) if root else download()
    return {
        int(folder.name): [load_recording(f) for f in sorted(folder.glob("*.txt"))]
        for folder in sorted(root.iterdir())
        if folder.is_dir()
    }
