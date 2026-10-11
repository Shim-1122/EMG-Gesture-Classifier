"""Signal conditioning: filtering, envelope extraction, and windowing.

Filters are causal (sosfilt, not zero-phase sosfiltfilt) so the pipeline only
uses past samples, matching what a real-time controller can compute.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt

from .data import FS, GESTURES


def highpass(emg: np.ndarray, cutoff_hz: float = 5.0, fs: int = FS, order: int = 4) -> np.ndarray:
    """Remove DC offset and slow motion artifact. Works on (n_samples, n_channels) arrays."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")
    return sosfilt(sos, emg, axis=0)


def envelope(emg: np.ndarray, cutoff_hz: float = 3.0, fs: int = FS, order: int = 2) -> np.ndarray:
    """Linear envelope: full-wave rectify, then low-pass. Approximates muscle activation level."""
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    return sosfilt(sos, np.abs(emg), axis=0)


def make_windows(
    emg: np.ndarray,
    labels: np.ndarray,
    win_ms: int = 200,
    step_ms: int = 50,
    fs: int = FS,
    classes=tuple(GESTURES),
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a recording into overlapping analysis windows.

    Keeps only windows that sit entirely inside one gesture in `classes`, so every
    window has one unambiguous label (transitions and unlabeled data are dropped).

    Returns
    -------
    windows : (n_windows, n_channels, win_len) array
    y       : (n_windows,) gesture label per window
    """
    win_len, step = int(win_ms * fs / 1000), int(step_ms * fs / 1000)

    # Zero-copy view of all windows (no Python loop)
    windows = sliding_window_view(emg, win_len, axis=0)[::step]     # (n, channels, win_len)
    window_labels = sliding_window_view(labels, win_len)[::step]    # (n, win_len)

    single_gesture = window_labels.min(axis=1) == window_labels.max(axis=1)
    wanted_class = np.isin(window_labels[:, 0], classes)
    keep = single_gesture & wanted_class
    return windows[keep], window_labels[keep, 0]
