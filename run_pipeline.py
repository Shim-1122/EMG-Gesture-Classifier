"""End-to-end EMG gesture recognition.

    python run_pipeline.py

1. Load all 36 subjects (downloads the data on first run), resample to 200 Hz
2. High-pass filter, cut 200 ms windows, extract Hudgins features
3. Evaluate three classifiers two ways:
     within-subject : train on session 1, test on session 2 (same person, new session)
     cross-subject  : train on some people, test on people never seen (6 subject folds)
4. Save figures to figures/ and a results table to results.md
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # file output only
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from emg import FS, GESTURES, N_CHANNELS, envelope, extract_features, highpass, load_all, make_windows

FIG_DIR = Path("figures")
WINDOW_LENGTHS_MS = [50, 100, 150, 200, 250, 300]
MODELS = {
    "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10)),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
}


# ---------------------------------------------------------------- evaluation
def featurize(session, win_ms=200):
    """(emg, labels) session -> (features, window labels)."""
    emg, labels = session
    windows, y = make_windows(highpass(emg), labels, win_ms=win_ms)
    return extract_features(windows), y


def within_subject(feats, model):
    """Per subject: fit on session 1, score on session 2.
    Returns per-subject accuracies and pooled (y_true, y_pred) for the confusion matrix."""
    accs, y_true, y_pred = [], [], []
    for (X_train, y_train), (X_test, y_test) in feats.values():
        pred = clone(model).fit(X_train, y_train).predict(X_test)
        accs.append(np.mean(pred == y_test))
        y_true.append(y_test)
        y_pred.append(pred)
    return np.array(accs), np.concatenate(y_true), np.concatenate(y_pred)


def cross_subject(feats, model, n_folds=6):
    """Subject-level folds: no person appears in both training and test data.

    Subjects are assigned to folds round-robin by sorted ID (fold = rank % n_folds),
    so the split is fixed and does not depend on the scikit-learn version.
    """
    X = np.vstack([X for sessions in feats.values() for X, _ in sessions])
    y = np.concatenate([y for sessions in feats.values() for _, y in sessions])
    fold_of = {subj: rank % n_folds for rank, subj in enumerate(sorted(feats))}
    fold = np.concatenate([np.full(len(y_s), fold_of[subj]) for subj, sessions in feats.items() for _, y_s in sessions])

    accs = []
    for k in range(n_folds):
        train, test = fold != k, fold == k
        accs.append(clone(model).fit(X[train], y[train]).score(X[test], y[test]))
    return np.array(accs)


# ------------------------------------------------------------------- figures
def shade_gestures(ax, t, labels):
    """Draw a colored band behind each labeled gesture segment."""
    edges = np.flatnonzero(np.diff(labels)) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        g = labels[start]
        if g in GESTURES:
            ax.axvspan(t[start], t[stop - 1], color=plt.cm.tab10(g), alpha=0.15, lw=0)


def plot_pipeline_stages(session, channel=4, t_max=33):
    emg, labels = session
    n = int(t_max * FS)
    emg, labels = emg[:n], labels[:n]
    t = np.arange(n) / FS
    filtered = highpass(emg)
    env = envelope(filtered)

    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    uv = 1e6  # volts -> microvolts
    axes[0].plot(t, emg[:, channel] * uv, lw=0.5, color="0.3")
    axes[0].set_title(f"1. Raw EMG (Channel {channel + 1})")
    axes[1].plot(t, filtered[:, channel] * uv, lw=0.5, color="C0")
    axes[1].set_title("2. High-Pass Filtered (5 Hz, Causal Butterworth)")
    axes[2].plot(t, np.abs(filtered[:, channel]) * uv, lw=0.4, color="0.6", label="rectified")
    axes[2].plot(t, env[:, channel] * uv, lw=1.8, color="C3", label="envelope")
    axes[2].set_title("3. Rectified Signal and Linear Envelope")
    axes[2].legend(loc="upper right")
    for ax in axes[:3]:
        ax.set_ylabel("µV")
        shade_gestures(ax, t, labels)

    # Each gesture activates a different subset of forearm muscles
    env_norm = env / env.max(axis=0)
    axes[3].imshow(env_norm.T, aspect="auto", cmap="magma", extent=[0, t[-1], N_CHANNELS + 0.5, 0.5])
    for edge in np.flatnonzero(np.diff(labels)) + 1:
        axes[3].axvline(t[edge], color="white", lw=0.8, ls=":")
    axes[3].set_title("4. Normalized Envelopes, All 8 Channels")
    axes[3].set_ylabel("channel")
    axes[3].set_xlabel("time (s)")

    # Label each gesture once, on the top panel
    edges = np.flatnonzero(np.diff(labels)) + 1
    for start, stop in zip(np.r_[0, edges], np.r_[edges, len(labels)]):
        if labels[start] in GESTURES:
            axes[0].text(t[(start + stop) // 2], axes[0].get_ylim()[1] * 0.85, GESTURES[labels[start]],
                         ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pipeline_stages.png", dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred):
    names = list(GESTURES.values())
    cm = confusion_matrix(y_true, y_pred, labels=list(GESTURES), normalize="true") * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", color="white" if cm[i, j] > 60 else "black")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("LDA Confusion Matrix, Within-Subject (% of Windows)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_summary(results, sweep):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    names = list(results)
    x = np.arange(len(names))
    for offset, key, label in [(-0.2, "within", "within-subject"), (0.2, "cross", "cross-subject")]:
        means = [results[m][key].mean() * 100 for m in names]
        stds = [results[m][key].std() * 100 for m in names]
        bars = ax1.bar(x + offset, means, 0.4, yerr=stds, capsize=4, label=label)
        ax1.bar_label(bars, fmt="%.1f", padding=12, fontsize=9)
    ax1.axhline(100 / len(GESTURES), ls="--", color="0.5", lw=1, label="chance (1 in 6)")
    ax1.set_xticks(x, names)
    ax1.set_ylabel("accuracy (%)")
    ax1.set_ylim(0, 125)
    ax1.set_title("Classifier Accuracy (Mean ± SD)")
    ax1.legend(loc="upper center", ncol=3, frameon=False, fontsize=9)

    means = np.array([a.mean() for a in sweep.values()]) * 100
    stds = np.array([a.std() for a in sweep.values()]) * 100
    ax2.plot(WINDOW_LENGTHS_MS, means, "o-", color="C0")
    ax2.fill_between(WINDOW_LENGTHS_MS, means - stds, means + stds, alpha=0.2)
    ax2.set_xlabel("window length (ms)")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Accuracy vs. Window Length (LDA, Within-Subject)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_summary.png", dpi=150)
    plt.close(fig)


def write_results(results, sweep, path="results.md"):
    lines = ["| Classifier | Within-Subject (%) | Cross-Subject (%) |", "|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['within'].mean()*100:.1f} ± {r['within'].std()*100:.1f} "
                     f"| {r['cross'].mean()*100:.1f} ± {r['cross'].std()*100:.1f} |")
    lines += ["", "| Window (ms) | LDA Within-Subject (%) |", "|---|---|"]
    lines += [f"| {w} | {a.mean()*100:.1f} ± {a.std()*100:.1f} |" for w, a in sweep.items()]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    print("Loading data ...")
    data = load_all()
    print(f"{len(data)} subjects loaded")

    feats = {subj: [featurize(s) for s in sessions] for subj, sessions in data.items()}

    results = {}
    for name, model in MODELS.items():
        print(f"Evaluating {name} ...")
        within, y_true, y_pred = within_subject(feats, model)
        results[name] = {"within": within, "cross": cross_subject(feats, model)}
        if name == "LDA":
            plot_confusion(y_true, y_pred)

    print("Sweeping window length ...")
    sweep = {}
    for win_ms in WINDOW_LENGTHS_MS:
        feats_w = {subj: [featurize(s, win_ms) for s in sessions] for subj, sessions in data.items()}
        sweep[win_ms] = within_subject(feats_w, MODELS["LDA"])[0]

    plot_pipeline_stages(data[1][0])
    plot_summary(results, sweep)
    write_results(results, sweep)
    print(f"Figures saved to {FIG_DIR}/")
