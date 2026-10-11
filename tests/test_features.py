"""Features are checked against closed-form values for a pure sine wave."""
import numpy as np
import pytest

from emg.features import extract_features, mav, rms, ssc, wl, zc

AMPLITUDE, FREQ_HZ, FS, N_CYCLES = 2.0, 10.0, 1000, 5
t = np.arange(int(N_CYCLES * FS / FREQ_HZ)) / FS
SINE = AMPLITUDE * np.sin(2 * np.pi * FREQ_HZ * t + 0.1)   # phase offset avoids samples landing exactly on 0
WINDOWS = SINE[None, None, :]                                # shape (1 window, 1 channel, n samples)


def test_mav_of_sine_is_2A_over_pi():
    assert mav(WINDOWS).item() == pytest.approx(2 * AMPLITUDE / np.pi, rel=1e-3)


def test_rms_of_sine_is_A_over_root2():
    assert rms(WINDOWS).item() == pytest.approx(AMPLITUDE / np.sqrt(2), rel=1e-3)


def test_waveform_length_is_4A_per_cycle():
    assert wl(WINDOWS).item() == pytest.approx(4 * AMPLITUDE * N_CYCLES, rel=1e-2)


def test_zero_crossings_twice_per_cycle():
    assert zc(WINDOWS).item() == 2 * N_CYCLES


def test_slope_sign_changes_twice_per_cycle():
    assert ssc(WINDOWS).item() == 2 * N_CYCLES


def test_thresholds_ignore_small_noise():
    noise = 1e-3 * np.random.default_rng(0).standard_normal((1, 1, 500))
    assert zc(noise, threshold=0.1).item() == 0
    assert ssc(noise, threshold=0.1).item() == 0


def test_feature_matrix_shape_and_finite():
    windows = np.random.default_rng(1).standard_normal((32, 8, 40))
    features = extract_features(windows)
    assert features.shape == (32, 5 * 8)
    assert np.all(np.isfinite(features))


def test_flat_window_does_not_produce_nan():
    assert np.all(np.isfinite(extract_features(np.zeros((1, 8, 40)))))
