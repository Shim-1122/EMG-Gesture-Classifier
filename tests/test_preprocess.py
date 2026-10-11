import numpy as np

from emg.data import load_recording
from emg.preprocess import envelope, highpass, make_windows

FS = 200


def test_highpass_removes_dc_offset():
    t = np.arange(10 * FS) / FS
    signal = 5.0 + np.sin(2 * np.pi * 40 * t)                  # 40 Hz tone on a large DC offset
    out = highpass(signal[:, None])[FS:]                       # skip the 1 s filter start-up transient
    assert abs(out.mean()) < 1e-2
    assert abs(np.std(out) - np.std(np.sin(2 * np.pi * 40 * t))) < 0.05  # the 40 Hz tone passes through


def test_envelope_tracks_amplitude():
    t = np.arange(10 * FS) / FS
    burst = np.where((t > 3) & (t < 6), 1.0, 0.0) * np.sin(2 * np.pi * 50 * t)
    env = envelope(burst[:, None])[:, 0]
    assert env[int(5 * FS)] > 10 * env[int(1 * FS)]            # active burst >> quiet baseline


def test_make_windows_keeps_only_pure_labeled_windows():
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    emg = np.random.default_rng(0).standard_normal((300, 8))
    windows, y = make_windows(emg, labels, win_ms=200, step_ms=50, fs=FS)   # 40-sample windows
    assert windows.shape[1:] == (8, 40)
    assert set(np.unique(y)) == {1, 2}                         # class 0 (unlabeled) is dropped
    # Check that every kept window really contains a single label
    starts = np.arange(0, 300 - 40 + 1, 10)
    pure = [s for s in starts if len(set(labels[s:s + 40])) == 1 and labels[s] != 0]
    assert len(y) == len(pure)


def test_load_recording_resamples_to_uniform_grid(tmp_path):
    # Tiny fake file in the dataset's format: irregular ms timestamps, 8 channels, class column
    t_ms = np.array([0, 1, 3, 4, 8, 9, 10, 15, 16, 20])
    cls = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    rows = np.column_stack([t_ms, np.tile(t_ms[:, None] * 1e-3, 8), cls])
    path = tmp_path / "fake.txt"
    header = "time\t" + "\t".join(f"channel{i}" for i in range(1, 9)) + "\tclass"
    np.savetxt(path, rows, delimiter="\t", header=header, comments="")

    emg, labels = load_recording(path, fs=200)                # 5 ms grid: 0, 5, 10, 15 ms
    assert emg.shape == (4, 8)
    np.testing.assert_allclose(emg[:, 0], [0, 5e-3, 10e-3, 15e-3])   # linear interpolation
    np.testing.assert_array_equal(labels, [0, 1, 2, 2])               # zero-order hold, never interpolated
