# Notebooks (run on free Colab/Kaggle)

Each notebook is **self-contained**: it clones the repo, mounts Drive, and rebuilds the manifest with
`seed=13` (so splits are identical everywhere) and imports everything from `src/`. Persistent state
lives on Drive under `MyDrive/faceproof/` (`data/crops`, `data/manifest.csv`, `models/features`,
`models/probes`, `reports/results.csv`, `reports/figures`). Set `BRANCH = "main"` in cell 1 once the
PR is merged. Run in order:

| # | Notebook | Day | Produces | Gate |
|---|----------|-----|----------|------|
| 01 | `01_preprocess_and_leakage.ipynb` | 1–2 | crops, manifest, leakage acc | matched batch loads; leakage ≈ chance |
| 02 | `02_cache_features.ipynb` | 2–3 | CLIP + ResNet features (Drive) | features cached + aligned |
| 03 | `03_probe_crossgen.ipynb` | 2–3 | C1/C4 probes, gap table | in-dist AUROC > 0.9; gap table (RQ1) |
| 04 | `04_robustness.ipynb` | 4 | AUROC vs corruption | JPEG curve for C1,C4 (RQ2) |
| 05 | `05_multigenerator.ipynb` | 5 | per-generator AUROC | ≥2 unseen generators |
| 06 | `06_calibration.ipynb` | 6 | ECE, reliability, temperature | AUROC\|EER\|ECE table (RQ3) |
| 07 | `07_seeds_ci.ipynb` | 7 | multi-seed + bootstrap CIs | core results with uncertainty |
| 08 | `08_extensions_c2_c3.ipynb` | 8 | C2 fine-tuned, C3 frequency | all 4 conditions in table |
| 09 | `09_aggregate_results.ipynb` | 9 | master tables (markdown) | numbers frozen |
| 10 | `10_figures.ipynb` | 10 | report figures (PNG) | 4 figures saved |

Then: `demo/app.py` (Day 11, Gradio/HF Spaces) and `reports/report.md` (Days 12–13).

> Keep heavy training on Kaggle (P100, 30 GPU-hrs/week) or Colab GPU. Notebooks 02/04/05/08 need GPU;
> 03/06/07/09/10 are CPU-light (they reuse cached features). See `PLAN.md` for the full day plan.
