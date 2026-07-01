"""
Days 3-6 — Linear probe on cached features (conditions C1 and C4).

A logistic-regression probe on frozen features is the whole 'detector' for the core
conditions: no backbone training, fits in seconds. Returns probabilities so we can
also evaluate calibration (Day 6).

Usage:
    from src.probe import train_probe, predict_proba
    clf = train_probe(X_train, y_train)
    p = predict_proba(clf, X_test)      # P(synthetic)
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_probe(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0,
                max_iter: int = 1000, seed: int = 13):
    """Standardize + logistic regression. Fit on TRAIN features only."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=max_iter, random_state=seed),
    )
    clf.fit(X_train, y_train)
    return clf


def predict_proba(clf, X: np.ndarray) -> np.ndarray:
    """Return P(class=1=synthetic). Assumes a binary probe (real vs synthetic)."""
    proba = clf.predict_proba(X)
    assert proba.shape[1] == 2, "predict_proba expects a binary (2-class) probe"
    return proba[:, 1]
