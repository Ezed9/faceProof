"""Unit tests for src/metrics.py — run with: python -m pytest tests/ -q

These exist because two real edge-case bugs were found here in review:
(1) ECE silently dropped predictions of exactly 0.0 (first-bin left edge was exclusive);
(2) bootstrap_ci crashed opaquely when every resample was single-class.
"""
import numpy as np
import pytest

from src.metrics import auroc, bootstrap_ci, eer, expected_calibration_error


def test_ece_counts_probability_zero():
    """Regression: p == 0.0 must be binned. All-synthetic labels predicted 0.0 => ECE == 1.0.
    Under the old bug every sample was dropped and this returned 0.0."""
    y = np.ones(100)
    p = np.zeros(100)
    assert expected_calibration_error(y, p) == pytest.approx(1.0)


def test_ece_perfectly_calibrated_is_zero():
    """Bins where mean predicted prob == empirical positive rate contribute nothing."""
    y = np.array([0] * 90 + [1] * 10 + [1] * 90 + [0] * 10)
    p = np.array([0.1] * 100 + [0.9] * 100)
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ece_hand_computed():
    """Single occupied bin: conf 0.8, frac_pos 0.5, full mass -> ECE = 0.3."""
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.8, 0.8, 0.8])
    assert expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.3)


def test_auroc_and_eer_extremes():
    y = np.array([0] * 50 + [1] * 50)
    perfect = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    assert auroc(y, perfect) == pytest.approx(1.0)
    assert eer(y, perfect) == pytest.approx(0.0)
    # inverted scores -> AUROC 0 (the below-chance regime this project reports)
    assert auroc(y, 1 - perfect) == pytest.approx(0.0)


def test_bootstrap_ci_sane_on_separable_data():
    rng = np.random.default_rng(0)
    y = np.array([0] * 300 + [1] * 300)
    s = y + rng.normal(0, 0.3, size=600)
    point, lo, hi = bootstrap_ci(y, s, n=200, seed=13)
    assert 0.5 < lo <= point <= hi <= 1.0


def test_bootstrap_ci_raises_cleanly_on_single_class():
    """Single-class input must raise ValueError (sklearn's point estimate or our resample guard),
    never an opaque IndexError from percentile-of-empty-list."""
    y = np.ones(50)
    s = np.linspace(0, 1, 50)
    with pytest.raises(ValueError):
        bootstrap_ci(y, s, n=20, seed=13)
