"""
Evaluation metrics: AUROC, EER, ECE (+ a simple bootstrap CI helper).
Used from Day 3 (AUROC/EER) and Day 6 (ECE/calibration).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y_true, y_score) -> float:
    return float(roc_auc_score(y_true, y_score))


def eer(y_true, y_score) -> float:
    """Equal Error Rate: the point where false-accept rate == false-reject rate."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE: weighted gap between confidence and accuracy across probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob > lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = (y_true[m] == (y_prob[m] >= 0.5)).mean()
        ece += (m.sum() / len(y_prob)) * abs(conf - acc)
    return float(ece)


def bootstrap_ci(y_true, y_score, metric_fn=auroc, n: int = 2000, seed: int = 13):
    """Percentile bootstrap CI over images. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = metric_fn(y_true, y_score)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            continue
        vals.append(metric_fn(y_true[b], y_score[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)
