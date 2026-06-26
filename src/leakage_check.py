"""
Day 2 — Leakage sanity check.

Idea: if a TRIVIAL frequency-statistics classifier (2D-DCT log-spectrum -> linear SVM)
can already separate your real vs fake crops, then your preprocessing is leaking a
shortcut (e.g. mismatched compression/resolution) and any later 'detector' result is
suspect. Run this on ~500 images PER CLASS before trusting anything.

Interpretation:
  - Accuracy near chance (~50%): good — no obvious shortcut. Proceed.
  - Accuracy high (e.g. >80-90%): BAD — fix compression/resolution matching first.

Usage:
    from src.leakage_check import run_leakage_check
    run_leakage_check("data/crops/real", "data/crops/stylegan", n_per_class=500)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dct_log_features(path: str | Path, size: int = 128, keep: int = 32) -> np.ndarray | None:
    """Grayscale -> 2D DCT -> log magnitude -> keep top-left low-freq block, flattened."""
    try:
        img = Image.open(path).convert("L").resize((size, size), Image.BICUBIC)
    except Exception:
        return None
    x = np.asarray(img, dtype=np.float32) / 255.0
    d = dctn(x, norm="ortho")
    d = np.log(np.abs(d) + 1e-8)
    return d[:keep, :keep].flatten()


def _load_class(folder: str | Path, n: int, size: int, keep: int):
    files = [p for p in sorted(Path(folder).rglob("*")) if p.suffix.lower() in _IMG_EXT][:n]
    feats = [f for f in (dct_log_features(p, size, keep) for p in files) if f is not None]
    return np.array(feats)


def run_leakage_check(real_dir: str | Path, fake_dir: str | Path, n_per_class: int = 500,
                      size: int = 128, keep: int = 32) -> float:
    """Return 5-fold CV accuracy of a trivial DCT+SVM real/fake classifier."""
    Xr = _load_class(real_dir, n_per_class, size, keep)
    Xf = _load_class(fake_dir, n_per_class, size, keep)
    X = np.vstack([Xr, Xf])
    y = np.array([0] * len(Xr) + [1] * len(Xf))

    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=5000))
    acc = cross_val_score(clf, X, y, cv=5).mean()

    print(f"[leakage] DCT+SVM 5-fold accuracy = {acc:.3f} "
          f"(real={len(Xr)}, fake={len(Xf)})")
    if acc > 0.80:
        print("  ⚠️  LEAKAGE LIKELY — fix compression/resolution matching before any model claim.")
    else:
        print("  ✅ No obvious shortcut. Safe to proceed.")
    return acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_dir", required=True)
    ap.add_argument("--fake_dir", required=True)
    ap.add_argument("--n_per_class", type=int, default=500)
    a = ap.parse_args()
    run_leakage_check(a.real_dir, a.fake_dir, a.n_per_class)
