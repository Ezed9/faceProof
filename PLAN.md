# FaceProof — Master day-by-day execution plan

> Operational companion to `CLAUDE.md` (scope/rules) and `HYPOTHESES.md` (RQs). Every day has an
> **objective**, **steps**, the **artifact** (notebook/file), and a **gate** that must pass before
> moving on. Cut from the bottom if behind (cut order in `CLAUDE.md`). Honesty rules are never cut.

## Conventions used by every experiment notebook
- **Self-contained:** each notebook clones the repo, mounts Drive, and rebuilds the manifest with
  `seed=13`, so splits are identical everywhere. Features are extracted in manifest row-order, so
  features and labels stay aligned.
- **Persistent storage (Drive):**
  `MyDrive/faceproof/` → `data/crops/`, `data/manifest.csv`, `models/features/*.npy`,
  `models/probes/*.joblib`, `reports/results.csv`, `reports/figures/`.
- **One results table:** every notebook appends rows to `reports/results.csv` with columns
  `condition,train_gen,test_gen,corruption,strength,seed,metric,value`. Day 9–10 read only this file.
- **Log every run** in `experiments.md` (daily 3-liner + a table row).

## Conditions
| Cond. | Detector | Tier | Built in |
|-------|----------|------|----------|
| C1 | Frozen ImageNet ResNet-50 + logistic probe | core | nb 02→03 |
| C4 | Frozen CLIP ViT-L/14 + logistic probe | core | nb 02→03 |
| C2 | Fine-tuned ResNet-50 (end-to-end) | extension | nb 08 |
| C3 | DCT log-spectrum + linear SVM | extension | nb 08 |

---

## WEEK 1 — core results

### Day 1 — setup + data + preprocessing ✅ DONE
- **Artifact:** `notebooks/01_preprocess_and_leakage.ipynb` (§1–7).
- **Steps:** download 140k + SFHQ Part 2 → compression-matched 224×224 crops → manifest + leakage-safe splits.
- **Gate:** ✅ matched real/fake batch loads `(16,3,224,224)` in `[0,1]`.

### Day 2 — leakage check + cache features + C4 in-distribution ✅ BUILT (run on Colab)
- **Artifacts:** `01` §8 (leakage), `02_cache_features.ipynb`, `03_probe_crossgen.ipynb` §1–4.
- **Steps:** DCT+SVM leakage check; cache CLIP (768-d) + ResNet (2048-d) features; train C4 probe; in-dist AUROC.
- **Gate:** leakage ≈ chance (<~0.80) **AND** in-distribution AUROC > ~0.9.

### Day 3 — C1 probe + cross-generator (RQ1)
- **Artifact:** `notebooks/03_probe_crossgen.ipynb` (full).
- **Steps:** train C1 (ResNet) and C4 (CLIP) probes on `train`; evaluate **in-dist** (held-out StyleGAN)
  and **cross-gen** (unseen SD `test` split); save probes to `models/probes/`; write the
  **generalization-gap table** (AUROC/EER, in-dist vs unseen, per condition) to `results.csv`.
- **Gate:** generalization-gap table exists; H1 directional check (CLIP gap ≤ ResNet gap).

### Day 4 — robustness sweep (RQ2)
- **Artifact:** `notebooks/04_robustness.ipynb`.
- **Steps:** for each corruption strength (`configs/experiment.yaml`), corrupt the **test images only**,
  re-extract CLIP/ResNet features, predict with the saved probes, log AUROC vs strength. JPEG is core;
  resize/blur/noise are extensions.
- **Gate:** JPEG AUROC-vs-quality curve for C1 and C4 in `results.csv`.

### Day 5 — multi-generator (≥2 unseen generators)
- **Artifact:** `notebooks/05_multigenerator.ipynb`.
- **Steps:** download SFHQ-T2I (SDXL/Flux/DALL-E, labelled in CSV) → preprocess → cache features →
  evaluate saved C1/C4 probes on each generator. Cheap optional: SFHQ Part 4 (SD 2.1).
- **Gate:** cross-gen AUROC/EER for ≥2 unseen generators in `results.csv`.

### Day 6 — calibration (RQ3)
- **Artifact:** `notebooks/06_calibration.ipynb` (uses `src/calibration.py`).
- **Steps:** compute ECE + reliability diagrams for C1/C4 on in-dist vs each unseen generator; fit
  temperature on `val`, re-measure ECE. Show AUROC unchanged, ECE moves.
- **Gate:** AUROC|EER|ECE table (pre/post temperature) for in-dist vs unseen — the headline finding.

### Day 7 — multi-seed + bootstrap CIs
- **Artifact:** `notebooks/07_seeds_ci.ipynb`.
- **Steps:** rebuild splits for `seeds:[13,37,71]`, retrain probes, log per-seed metrics; add
  percentile bootstrap CIs over images (`metrics.bootstrap_ci`) for the core numbers.
- **Gate:** core results reported with mean±spread and CIs.

---

## WEEK 2 — extensions, artifact, write-up

### Day 8 — C2 fine-tuned ResNet + C3 frequency baseline
- **Artifact:** `notebooks/08_extensions_c2_c3.ipynb`.
- **Steps:** C2 — fine-tune ResNet-50 end-to-end (config `model.yaml:finetune_resnet`), eval in-dist +
  cross-gen. C3 — DCT log-spectrum features (`leakage_check.dct_log_features`) → linear SVM detector,
  eval in-dist + cross-gen. Append to `results.csv`.
- **Gate:** all four conditions (C1–C4) present in the results table.

### Day 9 — finalize numbers (freeze)
- **Artifact:** `notebooks/09_aggregate_results.ipynb`.
- **Steps:** load `results.csv`, build the final master tables (generalization gap; robustness;
  calibration), sanity-check for gaps/typos, freeze. No new experiments after today.
- **Gate:** one clean master results table; numbers frozen.

### Day 10 — figures
- **Artifact:** `notebooks/10_figures.ipynb` → `reports/figures/`.
- **Steps:** generate (1) system diagram [static], (2) cross-generator bar/table, (3) robustness curves,
  (4) reliability diagrams — all from `results.csv`/saved predictions.
- **Gate:** all four report figures saved as PNG/PDF.

### Day 11 — Gradio demo + repo polish
- **Artifact:** `demo/app.py`, `demo/requirements.txt` (Hugging Face Spaces).
- **Steps:** load the saved CLIP probe + temperature; upload-a-face → real/synthetic + calibrated
  confidence + robustness note; optional attack-then-defend panel with pre-generated faces.
  Reproducibility pass (README run order, seeds, paths).
- **Gate:** demo runs locally and on Spaces; repo reproduces from README.

### Day 12 — report part 1
- **Artifact:** `reports/report.md` (§1–5: abstract, intro, contributions, related work, method).
- **Gate:** first half drafted from real method/config (no results invented).

### Day 13 — report part 2
- **Artifact:** `reports/report.md` (§6–10: experiments, results, threats, limitations, conclusion).
- **Gate:** results section filled from frozen `results.csv`; honest single-seed/CI statements.

### Day 14 — polish + slides + backup
- **Steps:** proofread, slide deck, record a backup demo video, buffer.
- **Gate:** submission-ready bundle.

---

## Dependency graph (what blocks what)
```
01 (crops) ─► 02 (features) ─► 03 (probes+gap) ─┬─► 04 (robustness)
                                                ├─► 06 (calibration)
                                                └─► 07 (seeds/CI)
05 (multi-gen) needs its own download → preprocess → features, then reuses 03 probes
08 (C2/C3) needs 01 crops (C2) and 02-style features/DCT (C3)
09 aggregates results.csv ─► 10 figures ─► 11 demo ; 12–13 report ; 14 polish
```
