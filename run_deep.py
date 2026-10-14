"""Deep learning comparison: a PyTorch 1D CNN on filtered EMG windows vs. the LDA baseline.

    python run_deep.py

Uses exactly the same data, 200 ms / 50 ms windows, and splits as run_pipeline.py:
    within-subject : train on session 1, test on session 2
    cross-subject  : 6 fixed subject folds (no person in both train and test)
The CNN sees the high-pass filtered window (8 channels x 40 samples); LDA sees the 40 Hudgins
features. Writes results_deep.md and figures/cnn_vs_lda.png. About 10 minutes on a laptop CPU.
"""
import subprocess
import sys
import time
from pathlib import Path

try:
    import torch
except ImportError:                                            # first run from the beginner launcher
    print("PyTorch is not installed yet; installing it (one time, a few hundred MB) ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-deep.txt"], check=True)
    import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from emg import highpass, load_all, make_windows
from emg.deep import CNNClassifier
from run_pipeline import FIG_DIR, MODELS, cross_subject, featurize, within_subject

WITHIN_EPOCHS, CROSS_EPOCHS = 30, 20


def windows(session, win_ms=200):
    emg, labels = session
    return make_windows(highpass(emg), labels, win_ms=win_ms)


def latency_ms(model, sample, n=500):
    """Mean single-window inference time on this CPU (one thread), the way a controller would call it."""
    torch.set_num_threads(1)
    model.predict(sample)                                     # warm-up
    t = time.perf_counter()
    for _ in range(n):
        model.predict(sample)
    return (time.perf_counter() - t) / n * 1000


def plot(results, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names, x = list(results), np.arange(2)
    for i, name in enumerate(names):
        means = [results[name][k].mean() * 100 for k in ("within", "cross")]
        stds = [results[name][k].std() * 100 for k in ("within", "cross")]
        bars = ax.bar(x + (i - 0.5) * 0.38, means, 0.38, yerr=stds, capsize=4, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=12, fontsize=9)
    ax.set_xticks(x, ["within-subject\n(new session)", "cross-subject\n(new person)"])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 115)
    ax.set_title("1D CNN (PyTorch) vs. LDA, Same Splits")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads()))
    FIG_DIR.mkdir(exist_ok=True)
    print("Loading data ...")
    data = load_all()
    wins = {s: [windows(x) for x in sessions] for s, sessions in data.items()}
    feats = {s: [featurize(x) for x in sessions] for s, sessions in data.items()}

    results = {"LDA (Hudgins features)": {}, "1D CNN (PyTorch)": {}}
    print("LDA ...")
    results["LDA (Hudgins features)"]["within"] = within_subject(feats, MODELS["LDA"])[0]
    results["LDA (Hudgins features)"]["cross"] = cross_subject(feats, MODELS["LDA"])
    print("CNN, within-subject (36 models) ...")
    results["1D CNN (PyTorch)"]["within"] = within_subject(wins, CNNClassifier(epochs=WITHIN_EPOCHS))[0]
    print("CNN, cross-subject (6 folds) ...")
    results["1D CNN (PyTorch)"]["cross"] = cross_subject(wins, CNNClassifier(epochs=CROSS_EPOCHS, batch_size=128))

    (X1, y1), _ = wins[sorted(wins)[0]]
    cnn = CNNClassifier(epochs=1).fit(X1, y1)
    lat = latency_ms(cnn, X1[:1])

    lines = ["| Model | Input | Within-Subject (%) | Cross-Subject (%) |", "|---|---|---|---|"]
    for name, inp in zip(results, ["40 Hudgins features", "filtered window, 8 x 40 samples"]):
        r = results[name]
        lines.append(f"| {name} | {inp} | {r['within'].mean()*100:.1f} ± {r['within'].std()*100:.1f} "
                     f"| {r['cross'].mean()*100:.1f} ± {r['cross'].std()*100:.1f} |")
    lines += ["", f"CNN: {cnn.n_parameters():,} parameters, {lat:.2f} ms per window on one CPU thread "
                  f"(window step is 50 ms)."]
    Path("results_deep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    plot(results, FIG_DIR / "cnn_vs_lda.png")
    print(f"Saved results_deep.md and {FIG_DIR / 'cnn_vs_lda.png'}")
