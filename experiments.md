# Experiment Log

Fill in one row per run. This is your research notebook — keep it current.

| Date | Run ID | Condition | Train gen | Test gen | Corruption | Seed | Metric | Value | Notes |
|------|--------|-----------|-----------|----------|------------|------|--------|-------|-------|
| 2026-06-?? | example | C4 (CLIP) | StyleGAN | StyleGAN (in-dist) | none | 13 | AUROC | 0.97 | sanity baseline |

## Daily log (3 lines/day: done / found / blocking)

### Day 1 (2026-06-26)
- Done: downloaded 140k real+fake (FFHQ real + StyleGAN fake) + SFHQ Part 2 (SD v1.4); ran compression-matched preprocessing via `src/preprocessing.py` (224×224 center-crop + JPEG-90 re-encode on all images); built manifest + leakage-safe splits (`src/data.py`); verified unseen generator (SD) is test-only; passed Day 1 gate — matched real/fake batch loads as (16, 3, 224, 224) float32 in [0, 1].
- Found: preprocessing config: `image_size=224`, `jpeg_match_quality=90`. Both raw dataset folders confirmed in Drive. Manifest written to `data/manifest.csv`. Split counts: train (StyleGAN train pool), val (15%), test_indist (15%), test (SD, unseen-only).
- Blocking: nothing — Day 2 (leakage check + CLIP feature extraction) is next.

### Day 2 (TODO — fill after running notebooks 01 leakage cell, 02, 03)
- Done: leakage check (DCT+SVM real vs StyleGAN, n=500/class) = ____ ; cached CLIP (768-d) + ResNet (2048-d) features to Drive; trained C4 probe, in-distribution AUROC = ____ .
- Found:
- Blocking:
