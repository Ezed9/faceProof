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
    """ECE for the positive-class probability: weighted gap between the mean predicted
    P(class=1) and the empirical positive rate, binned by predicted probability. This is
    self-consistent with a reliability diagram of predicted-prob (x) vs fraction-positive (y).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # First bin includes its left edge so predictions of exactly 0.0 are counted
        # (an unclipped sigmoid can underflow to 0.0 for very confident "real" calls).
        m = ((y_prob >= lo) if i == 0 else (y_prob > lo)) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()        # mean predicted P(class=1) in this bin
        frac_pos = y_true[m].mean()    # empirical fraction of positives in this bin
        ece += (m.sum() / len(y_prob)) * abs(conf - frac_pos)
    return float(ece)


def bootstrap_ci(y_true, y_score, metric_fn=auroc, n: int = 2000, seed: int = 13):
    """Percentile bootstrap CI over images. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = metric_fn(y_true, y_score)
    vals = []
    skipped = 0
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y_true[b])) < 2:
            skipped += 1          # metric undefined on a single-class resample
            continue
        vals.append(metric_fn(y_true[b], y_score[b]))
    if not vals:
        raise ValueError(
            "bootstrap_ci: every resample was single-class — eval set too small or too imbalanced for a CI"
        )
    if skipped > 0.05 * n:
        print(f"[bootstrap_ci] warning: {skipped}/{n} resamples skipped (single-class); CI from {len(vals)}")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)
