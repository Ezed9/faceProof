"""
Day 1-3 — Manifest + leakage-safe splits.

Build a single manifest (one row per image: path, label, generator) and split it so
no source/identity crosses train/val/test, and so each unseen generator is held out
for testing only. Keep this simple — for purely synthetic faces there is no real
identity to leak, but never let a real source scene appear in two splits.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_manifest(sources: dict, out_csv: str | None = None) -> pd.DataFrame:
    """sources: {name: {dir, label, generator}} -> DataFrame[path,label,generator,source]."""
    rows = []
    for name, spec in sources.items():
        d = Path(spec["dir"])
        if not d.exists():
            print(f"[data] WARNING: {d} not found (skipping {name})")
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in _IMG_EXT:
                rows.append({"path": str(p), "label": int(spec["label"]),
                             "generator": spec["generator"], "source": name})
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    print(f"[data] manifest: {len(df)} images across {df['generator'].nunique()} generators")
    return df


def make_splits(df: pd.DataFrame, train_generators, test_unseen, val_fraction=0.15,
                test_fraction=0.15, seed=13) -> pd.DataFrame:
    """Add a 'split' column. Unseen-generator rows are forced to test only."""
    df = df.copy()
    df["split"] = "unused"

    # Unseen generators -> test only.
    unseen_mask = df["generator"].isin(test_unseen)
    df.loc[unseen_mask, "split"] = "test"

    # Training pool: real + train generators.
    pool = df[(df["generator"].isin(train_generators)) | (df["label"] == 0)].copy()
    pool = pool.sample(frac=1.0, random_state=seed)
    n = len(pool)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    idx = pool.index.tolist()
    df.loc[idx[:n_test], "split"] = "test_indist"
    df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    df.loc[idx[n_test + n_val:], "split"] = "train"
    print(df["split"].value_counts())
    return df


def validate_splits(df: pd.DataFrame) -> bool:
    """Assert no source appears in more than one split (basic leakage guard)."""
    bad = df.groupby("source")["split"].nunique()
    leaks = bad[bad > 1]
    if len(leaks):
        print(f"[data] ⚠️ sources spanning splits: {list(leaks.index)}")
        return False
    print("[data] ✅ no source spans multiple splits")
    return True
