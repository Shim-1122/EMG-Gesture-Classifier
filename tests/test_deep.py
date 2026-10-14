"""Tests for the PyTorch CNN. Skipped when PyTorch is not installed (it is optional)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402

from emg.deep import CLASSES, CNNClassifier, EMGNet  # noqa: E402


def toy_windows(n_per_class=40, seed=0):
    """Each gesture activates a different channel at a different frequency, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(40) / 200
    X, y = [], []
    for k, g in enumerate(CLASSES):
        for _ in range(n_per_class):
            w = 0.3 * rng.standard_normal((8, 40))
            w[k] += np.sin(2 * np.pi * (20 + 10 * k) * t + rng.uniform(0, 2 * np.pi))
            X.append(w * 1e-5)                                   # EMG-scale amplitudes (volts)
            y.append(g)
    return np.array(X), np.array(y)


def test_forward_shape():
    out = EMGNet()(torch.zeros(5, 8, 40))
    assert out.shape == (5, len(CLASSES))


def test_model_is_small_enough_for_embedded_use():
    assert CNNClassifier().n_parameters() < 50_000


def test_learns_separable_gestures_and_returns_gesture_ids():
    X, y = toy_windows()
    X_test, y_test = toy_windows(seed=1)
    clf = CNNClassifier(epochs=15).fit(X, y)
    assert set(clf.predict(X_test)) <= set(CLASSES)
    assert clf.score(X_test, y_test) > 0.9


def test_deterministic_with_seed():
    X, y = toy_windows(n_per_class=10)
    a = CNNClassifier(epochs=2, seed=3).fit(X, y).predict(X)
    b = CNNClassifier(epochs=2, seed=3).fit(X, y).predict(X)
    assert np.array_equal(a, b)


def test_works_with_sklearn_clone():
    """run_pipeline.within_subject / cross_subject clone the model for every split."""
    c = clone(CNNClassifier(epochs=7, width=16))
    assert c.epochs == 7 and c.width == 16


def test_normalization_uses_training_statistics_only():
    X, y = toy_windows(n_per_class=10)
    clf = CNNClassifier(epochs=1).fit(X, y)
    assert clf.mean_.shape == (1, 8, 1) and clf.std_.shape == (1, 8, 1)
    assert np.allclose(clf.mean_, X.mean(axis=(0, 2), keepdims=True))
