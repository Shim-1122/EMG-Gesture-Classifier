"""1D convolutional network (PyTorch) that learns features directly from filtered EMG windows.

The Hudgins pipeline hand-designs 40 features per window; this model instead takes the
high-pass filtered window itself, shaped (8 channels, 40 samples), and learns its own
temporal filters. It is evaluated with exactly the same session and subject splits as the
LDA baseline in run_pipeline.py, so the numbers are directly comparable.

PyTorch is optional: install it with `pip install -r requirements-deep.txt`.
"""
import numpy as np
import torch
from torch import nn

from .data import GESTURES, N_CHANNELS

CLASSES = np.array(sorted(GESTURES))          # gesture ids 1..6 -> class indices 0..5


class EMGNet(nn.Module):
    """Small causal-friendly 1D CNN: three conv blocks, global average pooling, linear head.

    About 25k parameters, so per-window inference is cheap enough for an embedded controller.
    """

    def __init__(self, n_channels: int = N_CHANNELS, n_classes: int = len(CLASSES), width: int = 32,
                 dropout: float = 0.3):
        super().__init__()

        def block(c_in, c_out, k):
            return nn.Sequential(nn.Conv1d(c_in, c_out, k, padding=k // 2), nn.BatchNorm1d(c_out), nn.ReLU())

        self.features = nn.Sequential(
            block(n_channels, width, 7),
            block(width, 2 * width, 5),
            nn.MaxPool1d(2),
            block(2 * width, 2 * width, 3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

    def forward(self, x):                      # x: (batch, channels, samples)
        return self.head(self.features(x))


class CNNClassifier:
    """scikit-learn style wrapper (fit / predict / score) so the CNN drops into the same evaluation code."""

    def __init__(self, epochs: int = 40, batch_size: int = 64, lr: float = 3e-3, weight_decay: float = 1e-3,
                 width: int = 32, seed: int = 0):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.weight_decay, self.width, self.seed = weight_decay, width, seed

    def get_params(self, deep=True):
        return dict(epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
                    weight_decay=self.weight_decay, width=self.width, seed=self.seed)

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ----------------------------------------------------------------- data
    def _scale(self, windows):
        """Per-channel standardization, with statistics from the training set only."""
        return torch.as_tensor((windows - self.mean_) / self.std_, dtype=torch.float32)

    @staticmethod
    def _to_index(y):
        return np.searchsorted(CLASSES, y)

    # ------------------------------------------------------------------ API
    def fit(self, windows, y):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        windows = np.asarray(windows, dtype=np.float64)
        self.mean_ = windows.mean(axis=(0, 2), keepdims=True)
        self.std_ = windows.std(axis=(0, 2), keepdims=True) + 1e-12
        X, t = self._scale(windows), torch.as_tensor(self._to_index(y))

        self.model_ = EMGNet(windows.shape[1], len(CLASSES), self.width)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(X) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

        self.model_.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), self.batch_size):
                idx = order[start:start + self.batch_size]
                if len(idx) < 2:                 # BatchNorm needs more than one sample
                    continue
                xb = X[idx]
                xb = xb * (1 + 0.1 * torch.randn(xb.shape[0], xb.shape[1], 1))   # per-channel gain jitter
                opt.zero_grad()
                loss_fn(self.model_(xb), t[idx]).backward()
                opt.step()
                sched.step()
        return self

    @torch.no_grad()
    def predict(self, windows):
        self.model_.eval()
        logits = self.model_(self._scale(np.asarray(windows, dtype=np.float64)))
        return CLASSES[logits.argmax(dim=1).numpy()]

    def score(self, windows, y):
        return float(np.mean(self.predict(windows) == np.asarray(y)))

    def n_parameters(self):
        return sum(p.numel() for p in EMGNet(N_CHANNELS, len(CLASSES), self.width).parameters())
