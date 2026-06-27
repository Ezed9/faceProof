"""
Day 6 — Temperature scaling for calibration (RQ3).

Temperature scaling divides the model logits by a single learned scalar T > 0, fit on a
held-out set by minimizing negative log-likelihood. Because it is a monotonic transform of
the logits it CANNOT change ranking (AUROC/EER are unchanged) — it only rescales confidence,
i.e. it fixes ECE. T > 1 softens an over-confident model; T < 1 sharpens an under-confident one.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Find T > 0 minimizing NLL of sigmoid(logits / T). Fit on a held-out (val) set."""
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)

    def nll(log_t: float) -> float:
        t = np.exp(log_t)                       # parameterize as exp() to keep T > 0
        p = np.clip(_sigmoid(logits / t), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated P(class=1=synthetic) = sigmoid(logits / T)."""
    return _sigmoid(np.asarray(logits, dtype=np.float64) / temperature)
