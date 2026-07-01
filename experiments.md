# Experiment Log

Fill in one row per run. This is your research notebook — keep it current.

| Date | Run ID | Condition | Train gen | Test gen | Corruption | Seed | Metric | Value | Notes |
|------|--------|-----------|-----------|----------|------------|------|--------|-------|-------|
| 2026-06-28 | d3-c4-indist | C4 (CLIP)   | StyleGAN | StyleGAN (in-dist) | none      | 13 | AUROC | 0.997 | core, ~6k/class subset |
| 2026-06-28 | d3-c1-indist | C1 (ResNet) | StyleGAN | StyleGAN (in-dist) | none      | 13 | AUROC | 0.876 | weaker baseline |
| 2026-06-28 | d3-c4-sd     | C4 (CLIP)   | StyleGAN | SD (unseen)        | none      | 13 | AUROC | 0.907 | H1 metric (cross-gen) |
| 2026-06-28 | d3-c1-sd     | C1 (ResNet) | StyleGAN | SD (unseen)        | none      | 13 | AUROC | 0.833 | CLIP > ResNet → H1 ✅ |
| 2026-06-28 | d4-c4-sd-q10 | C4 (CLIP)   | StyleGAN | SD (unseen)        | JPEG q=10 | 13 | AUROC | ~0.82 | heavy compression |
| 2026-06-28 | d4-c1-sd-q10 | C1 (ResNet) | StyleGAN | SD (unseen)        | JPEG q=10 | 13 | AUROC | ~0.64 | near failure regime |
| 2026-06-28 | d6-c4-id-ece | C4 (CLIP)   | StyleGAN | StyleGAN (in-dist) | none      | 13 | ECE   | 0.020 | balanced; temp→0.009 |
| 2026-06-28 | d6-c4-sd-ece | C4 (CLIP)   | StyleGAN | SD (unseen)        | none      | 13 | ECE   | 0.239 | balanced; temp→0.226 (no fix) — H3 |
| 2026-06-28 | d6-c1-sd-ece | C1 (ResNet) | StyleGAN | SD (unseen)        | none      | 13 | ECE   | 0.203 | balanced; temp→0.053 |
| 2026-06-28 | d5-c4-sdxl   | C4 (CLIP)   | StyleGAN | SDXL (unseen)      | none      | 13 | AUROC | 0.310 | below chance — detector blind |
| 2026-06-28 | d5-c4-flux   | C4 (CLIP)   | StyleGAN | Flux (unseen)      | none      | 13 | AUROC | 0.411 | below chance |
| 2026-06-28 | d5-c4-dalle  | C4 (CLIP)   | StyleGAN | DALL-E3 (unseen)   | none      | 13 | AUROC | 0.343 | below chance; n=1123 |
| 2026-06-28 | d5-c1-sdxl   | C1 (ResNet) | StyleGAN | SDXL (unseen)      | none      | 13 | AUROC | 0.304 | below chance |
| 2026-06-28 | d7-c4-sd-3s  | C4 (CLIP)   | StyleGAN | SD (unseen)        | none      | 13,37,71 | AUROC | 0.920±0.015 | 95% CI [0.897,0.915] |
| 2026-06-28 | d7-c1-sd-3s  | C1 (ResNet) | StyleGAN | SD (unseen)        | none      | 13,37,71 | AUROC | 0.836±0.021 | 95% CI [0.818,0.848]; CIs disjoint → CLIP>ResNet sig |
| 2026-06-29 | d8-c2-id     | C2 (ft ResNet) | StyleGAN | StyleGAN (in-dist) | none   | 13 | AUROC | 0.997 | fine-tuned end-to-end |
| 2026-06-29 | d8-c2-sd     | C2 (ft ResNet) | StyleGAN | SD (unseen)        | none   | 13 | AUROC | 0.938 | beat frozen CLIP on SD |
| 2026-06-29 | d8-c3-id     | C3 (DCT+SVM)   | StyleGAN | StyleGAN (in-dist) | none   | 13 | AUROC | 0.704 | frequency baseline |
| 2026-06-29 | d8-c3-sd     | C3 (DCT+SVM)   | StyleGAN | SD (unseen)        | none   | 13 | AUROC | 0.630 | weak across shift |
| 2026-06-29 | d8-c2-sdxl   | C2 (ft ResNet) | StyleGAN | SDXL (unseen)      | none   | 13 | AUROC | 0.484 | below chance |
| 2026-06-29 | d8-c2-flux   | C2 (ft ResNet) | StyleGAN | Flux (unseen)      | none   | 13 | AUROC | 0.322 | below chance |
| 2026-06-29 | d8-c2-dalle  | C2 (ft ResNet) | StyleGAN | DALL-E3 (unseen)   | none   | 13 | AUROC | 0.170 | below chance; meanP_fake≈0 |
| 2026-06-29 | d8-c3-t2i    | C3 (DCT+SVM)   | StyleGAN | SDXL/Flux/DALL-E   | none   | 13 | AUROC | 0.35–0.37 | below chance |

## Daily log (3 lines/day: done / found / blocking)

### Day 1 (2026-06-26)
- Done: downloaded 140k real+fake (FFHQ real + StyleGAN fake) + SFHQ Part 2 (SD v1.4); ran compression-matched preprocessing via `src/preprocessing.py` (224×224 center-crop + JPEG-90 re-encode on all images); built manifest + leakage-safe splits (`src/data.py`); verified unseen generator (SD) is test-only; passed Day 1 gate — matched real/fake batch loads as (16, 3, 224, 224) float32 in [0, 1].
- Found: preprocessing config: `image_size=224`, `jpeg_match_quality=90`. Both raw dataset folders confirmed in Drive. Manifest written to `data/manifest.csv`. Split counts: train (StyleGAN train pool), val (15%), test_indist (15%), test (SD, unseen-only).
- Blocking: nothing — Day 2 (leakage check + CLIP feature extraction) is next.

### Day 2 (2026-06-27)
- Done: ran DCT+SVM leakage check (real vs StyleGAN, n=500/class) = ____ [confirm value from nb01 §8]; cached frozen CLIP (768-d) + ResNet (2048-d) features to Drive (`models/features/*.npy`); built/saved manifest with splits.
- Found: features align with manifest (asserts pass; CLIP in-dist AUROC 0.997 later confirms alignment). In-distribution gate: C4 CLIP 0.997 > 0.90 ✅; C1 ResNet 0.876 (weaker baseline, expected).
- Blocking: leakage value not recorded here — paste it from the notebook to close the Day 2 gate properly.

### Day 3 (2026-06-28) — cross-generator (RQ1/H1)
- Done: trained C1 + C4 logistic probes on `train` only (saved to `models/probes/`); evaluated in-dist (held-out StyleGAN) and cross-gen (unseen SD). Cross-gen AUROC: **C4 CLIP 0.907 vs C1 ResNet 0.833**.
- Found: **H1 supported** — frozen CLIP generalizes to the unseen generator better (higher cross-gen AUROC). The in-dist−cross-gen "gap" is larger for CLIP (0.090 vs 0.043) but that conflates CLIP's near-perfect in-dist; report cross-gen AUROC as the primary metric, gap as context.
- Blocking: single-seed, ~6k/class subset — Day 7 multi-seed + CIs needed before claiming the margin is robust.

### Day 4 (2026-06-28) — robustness JPEG sweep (RQ2/H2)
- Done: corrupted test images at JPEG q=[90,70,50,30,10], re-extracted features, scored saved probes; curve saved to `reports/figures/robustness_jpeg.png`.
- Found: **H2 confirmed** — all four conditions degrade as quality drops. CLIP more robust than ResNet at every level; graceful decline to ~q30 then a cliff at q≤10 (ResNet/SD → ~0.64, near failure). At q=10, CLIP cross-gen (~0.82) > ResNet in-dist (~0.69) — roles invert under stress.
- Blocking: low-quality tail is noisy (single seed; green CLIP/SD non-monotonic bump at q10 is noise, not a real effect).

### Day 5 (2026-06-28) — multi-generator (SFHQ-T2I: SDXL / Flux / DALL-E 3)
- Done: evaluated saved C1/C4 probes on 3 unseen T2I generators vs held-out reals. AUROC (C4 CLIP / C1 ResNet): SDXL 0.310 / 0.304; FLUX1_schnell 0.411 / 0.273; DALLE3 0.343 / 0.182 — **all below 0.5**.
- Found: **catastrophic generalization failure on modern T2I generators** — both detectors are blind/inverted (mean P(synthetic): reals 0.031 vs SDXL fakes 0.014 for CLIP; probe rates modern fakes as *more real than real*). Same probes/reals/preprocessing as Day 3 where SD v1.4 → 0.907, so the collapse is generator-driven (2022 diffusion detectable, 2023-24 T2I not). CLIP ≥ ResNet on every generator (H1 direction holds) but both unusable. Crops verified to be valid aligned faces.
- Blocking: single seed; DALL-E n=1123 (noisy). Threat to validity: reals 256px-origin vs T2I 1024px-origin (constant across Day 3/5, so not the cause). Optional MTCNN-align re-run for extra rigor.

### Day 6 (2026-06-28) — calibration (RQ3/H3), balanced 50/50 groups
- Done: ECE + reliability + temperature scaling on balanced in-dist vs unseen-SD groups (n≈1770/1830). ECE: C4 CLIP 0.020 (in-dist) → 0.239 (cross-gen); C1 ResNet 0.162 → 0.203. Temperature (fit on val) fixes in-dist for both (CLIP→0.009, ResNet→0.047); on cross-gen it FAILS for CLIP (0.239→0.226) but helps ResNet (0.203→0.053, T≈7.9).
- Found: **H3 supported** — CLIP near-perfectly calibrated in-distribution but ~12× worse on the unseen generator while keeping AUROC 0.89; in-dist temperature scaling does NOT repair CLIP's shift-induced miscalibration. Reliability: confident "real" calls on SD fakes (predicted≈0 → ~27% actually synthetic). Computed on BALANCED groups (cross-gen set was ~85% synthetic; imbalance had inflated raw ECE 0.39→0.24).
- Blocking: single seed — Day 7 bootstrap CIs next. ResNet's temp improvement reflects uniform over-confidence (T≈7.9) but it stays the weaker detector.

### Day 7 (2026-06-28) — multi-seed + bootstrap CIs
- Done: re-split + retrain probes over seeds [13,37,71]; bootstrap CIs (2000×, seed 13) on cross-gen AUROC. Cross-gen mean±std: C4 CLIP 0.920±0.015, C1 ResNet 0.836±0.021; in-dist C4 0.996±0.001, C1 0.872±0.004. Bootstrap 95% CI: CLIP [0.897,0.915], ResNet [0.818,0.848].
- Found: results seed-stable (small std). **CLIP > ResNet cross-gen is significant** — 95% CIs disjoint (CLIP lower 0.897 > ResNet upper 0.848). H1 upgraded from directional to statistically supported. Core results now locked with uncertainty.
- Blocking: only 3 seeds (±std is rough) but bootstrap corroborates. ✅

### Day 8 (2026-06-29) — extensions: C2 fine-tuned ResNet + C3 DCT frequency
- Done: C2 = ResNet-50 fine-tuned end-to-end; C3 = DCT log-spectrum + linear SVM (nb08 rebuilt to evaluate both on in-dist, SD, AND the three T2I generators). AUROC in-dist / SD: C2 0.996 / 0.942; C3 0.704 / 0.630. All four conditions now span every generator in results.csv.
- Found (SD): C2 (fine-tuned) generalized to SD as well as / better than frozen CLIP (0.942 vs 0.907) — "fine-tuning hurts generalization" did NOT hold on the moderate StyleGAN→SD shift. C3 frequency baseline weak (0.63).
- Found (T2I) — **decisive**: C2 AND C3 also collapse BELOW CHANCE on modern T2I. C2 AUROC SDXL 0.484 / Flux 0.322 / DALL-E3 0.170 (mean P(synthetic) on fakes ≈ 0.000 — ~100% sure they are real); C3 ≈ 0.35–0.37. **Conclusion:** no detector — frozen CLIP/ResNet, fine-tuned ResNet, or DCT frequency — survives modern text-to-image faces; fine-tuning does not rescue generalization. The failure is universal across detector design.

### Day 9 (2026-06-29) — aggregate + freeze numbers
- Done: built master tables from `results.csv` → `reports/table_auroc.md` (4 conditions × 5 generators) + `reports/table_calibration.md` (ECE before/after temp, C1/C4). All cells populated, no gaps.
- Found: master AUROC table is the headline — strong in-dist + SD (C2 0.942 ≥ C4 0.907 > C1 0.834 > C3 0.630), universal below-chance collapse across ALL detectors on ALL three T2I generators (0.17–0.48). CLIP SD = 0.907 (full-set, seed 13).
- Blocking: numbers FROZEN — no new experiments. Note: T2I "gap" columns not meaningful (below-chance = inverted); report raw T2I AUROC. ✅

### Day 10 (2026-06-29) — figures
- Done: generated 4 report figures → `reports/figures/` — system diagram, cross-generator AUROC bars, JPEG robustness curve, reliability diagrams. Fixed the bar-chart y-axis (was starting at 0.4 and clipping the below-chance T2I bars) → now 0–1 with a 0.5 chance line, generators ordered in-dist→T2I.
- Found: headline bar chart now shows the contrast clearly (green in-dist/SD vs all-below-chance T2I). Reliability diagrams show C4 CLIP miscalibrated on SD (above diagonal) vs near-diagonal in-dist — matches H3.
- Blocking: reliability figure axis labels cosmetic (say accuracy/confidence; data is fraction-positive vs predicted-prob). ✅
