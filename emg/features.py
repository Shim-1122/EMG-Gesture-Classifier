"""Hudgins time-domain features for myoelectric control.

Each function takes windows shaped (n_windows, n_channels, n_samples) and
reduces over the last axis, so all windows are processed in one vectorized call.

Reference: Hudgins, Parker & Scott, "A new strategy for multifunction
myoelectric control," IEEE Trans. Biomed. Eng., 1993.
"""
import numpy as np

FEATURES = ["log MAV", "log RMS", "log WL", "ZC", "SSC"]


def mav(w):
    """Mean absolute value: overall contraction intensity."""
    return np.mean(np.abs(w), axis=-1)


def rms(w):
    """Root mean square: signal power."""
    return np.sqrt(np.mean(w**2, axis=-1))


def wl(w):
    """Waveform length: cumulative path length (depends on amplitude and frequency)."""
    return np.sum(np.abs(np.diff(w, axis=-1)), axis=-1)


def zc(w, threshold: float = 0.0):
    """Zero crossings: frequency estimate. `threshold` rejects small noise-driven crossings."""
    sign_change = w[..., :-1] * w[..., 1:] < 0
    big_enough = np.abs(np.diff(w, axis=-1)) > threshold
    return np.sum(sign_change & big_enough, axis=-1)


def ssc(w, threshold: float = 0.0):
    """Slope sign changes: count of local peaks and valleys (frequency estimate)."""
    d = np.diff(w, axis=-1)
    return np.sum(-d[..., :-1] * d[..., 1:] > threshold, axis=-1)


def extract_features(windows: np.ndarray) -> np.ndarray:
    """Compute all 5 features on every channel -> (n_windows, 5 * n_channels).

    Amplitude features are log-transformed: EMG amplitude is roughly log-normal,
    and LDA assumes Gaussian class distributions.
    """
    eps = 1e-12  # avoids log(0) on flat-line windows
    return np.hstack([
        np.log(mav(windows) + eps),
        np.log(rms(windows) + eps),
        np.log(wl(windows) + eps),
        zc(windows),
        ssc(windows),
    ])
