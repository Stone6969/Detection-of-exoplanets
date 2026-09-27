"""Stage 4c - CLASSIFIER: multi-view 1-D CNN + scalar MLP, 4 classes,
trained on real labels + physics simulations with a partial-label loss.

Inputs  global view 1x201 | local views 7x61 | 23 scalar diagnostics
Output  logits for [planet, eclipsing_binary, blend, other]

Loss (novel part #5 - using every label the dataset actually gives):
  simulated samples   -> full 4-class cross-entropy (we know the class)
  real CP/KP planets  -> cross-entropy on class 0
  real FP/FA          -> "not a planet" but type unknown  ->  -log(1 - p_planet)
                         (partial label: any of EB/blend/other is accepted)
  real PC/APC         -> not used for training; scored at the end
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .views import LOCAL_CHANNELS, N_GLOBAL, N_LOCAL, SCALARS

N_CLASSES = 4
PARTIAL_NOT_PLANET = -2      # label code for real false positives


def _block(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                         nn.Conv1d(cout, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.ReLU(),
                         nn.MaxPool1d(2))


class AstroNetPlus(nn.Module):
    def __init__(self, n_scalars=len(SCALARS), n_local=len(LOCAL_CHANNELS), dropout=0.3):
        super().__init__()
        self.glob = nn.Sequential(_block(1, 16, 5), _block(16, 32, 5), _block(32, 64, 5),
                                  _block(64, 64, 5), nn.AdaptiveAvgPool1d(4), nn.Flatten())
        self.loc = nn.Sequential(_block(n_local, 32, 5), _block(32, 64, 5), _block(64, 64, 3),
                                 nn.AdaptiveAvgPool1d(4), nn.Flatten())
        self.sca = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * 4 + 64 * 4 + 64, 256), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(128, N_CLASSES))

    def forward(self, g, l, s):
        return self.head(torch.cat([self.glob(g), self.loc(l), self.sca(s)], 1))


def partial_label_loss(logits, y, weight=None):
    """y >= 0: normal class; y == PARTIAL_NOT_PLANET: any class except 0."""
    logp = F.log_softmax(logits, 1)
    full = y >= 0
    loss = torch.zeros(len(y), device=logits.device)
    if full.any():
        loss[full] = F.nll_loss(logp[full], y[full], weight=weight, reduction="none")
    part = y == PARTIAL_NOT_PLANET
    if part.any():
        loss[part] = -torch.logsumexp(logp[part][:, 1:], 1)     # -log(1 - p_planet)
    return loss.mean()


class Standardizer:
    """Per-feature mean/std for scalars (fitted on the training set only)."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return np.clip((X - self.mu) / self.sd, -8, 8).astype(np.float32)

    def state(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist())

    @classmethod
    def from_state(cls, d):
        s = cls()
        s.mu, s.sd = np.array(d["mu"]), np.array(d["sd"])
        return s


def augment(g, l):
    """Train-time augmentation: small phase shift, flip in time (transits are
    symmetric), amplitude jitter - mimics ephemeris errors of a real search."""
    if np.random.rand() < 0.5:
        g, l = g[..., ::-1].copy(), l[..., ::-1].copy()
    sh = np.random.randint(-2, 3)
    l = np.roll(l, sh, -1)
    g = np.roll(g, np.random.randint(-1, 2), -1)
    return g * np.random.uniform(0.9, 1.1), l
