# FaceProof — Auditing AI-Face Detectors as a Biometric Security Control

A reproducible study of **cross-generator generalization, robustness, and calibration** for
synthetic-face detection — framed as an enrollment-screening security control.

> **Honest scope:** this is a *directional reproduction* of CLIP-based fake-image detection
> (Ojha et al. 2023) in the faces domain, plus a reliability audit (robustness + calibration).
> It is **not** a novel architecture, an exact reproduction, or a state-of-the-art claim.

## Research questions & hypotheses
See [`HYPOTHESES.md`](HYPOTHESES.md). In short:
- **RQ1 / H1** — frozen CLIP features generalize to unseen generators better than frozen ImageNet features.
- **RQ2 / H2** — both detectors degrade under JPEG / real-world image corruption.
- **RQ3 / H3** — confidence calibration worsens under generator shift (over-confidence = a security liability).

## Detector conditions
| Cond. | Detector | Tier |
|-------|----------|------|
| C1 | Frozen ImageNet ResNet-50 + logistic probe | core |
| C4 | Frozen CLIP ViT-L/14 + logistic probe | core |
| C2 | Fine-tuned ResNet-50 (end-to-end) | extension |
| C3 | DCT log-spectrum + linear SVM (frequency baseline) | extension |

## Repository layout
```
faceproof/
├── configs/        # YAML configs (data, model, experiment) — no settings hard-coded
├── src/            # library code (preprocessing, features, probe, metrics, ...)
├── notebooks/      # Colab notebooks (run order in notebooks/README.md)
├── data/           # datasets + caches (gitignored)
├── models/         # checkpoints + cached features (gitignored)
├── reports/        # figures + the technical report
├── demo/           # Gradio demo (Hugging Face Spaces)
├── experiments.md  # running experiment log (fill in every run)
└── requirements.txt
```

## Datasets (all free — NOT committed to git)
| Role | Dataset | Generator |
|------|---------|-----------|
| Real | FFHQ / real split of "140k real and fake faces" | real |
| Train synthetic | 140k real and fake faces | StyleGAN |
| Unseen test 1 | SFHQ (Parts 2/4) | Stable Diffusion |
| Unseen tests 2–3 | SFHQ-T2I | SDXL, Flux, DALL-E 3 |

- 140k: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- SFHQ: https://github.com/SelfishGene/SFHQ-dataset
- SFHQ-T2I: https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models

Place/download datasets under `data/raw/` (see `configs/data.yaml`). Respect each dataset's license
(FFHQ is CC BY-NC-SA — research use only).

## Setup
```bash
pip install -r requirements.txt
```
Recommended: run on free Google Colab (T4) or Kaggle (P100). Cache features to Drive once to save GPU time.

## Run order (2-week plan)
1. `src/preprocessing.py` — align/crop faces + compression matching  *(Day 1)*
2. `src/leakage_check.py` — DCT leakage sanity check  *(Day 2)*
3. `src/features.py` — cache CLIP & ResNet features  *(Days 2–3)*
4. `src/probe.py` + `src/metrics.py` — train probes, evaluate cross-generator  *(Days 3–6)*

Full schedule: see `../Execution_Plan_2weeks.md`.

## Status
**Experiments complete (Days 1–10); report + demo in progress (Days 11–14).** All four detector
conditions evaluated across five generators, with a robustness sweep, calibration audit, multi-seed
runs, bootstrap CIs, and figures. Full log: [`experiments.md`](experiments.md); write-up:
[`reports/report.md`](reports/report.md); reproduction setup: [`SETUP.md`](SETUP.md).

### Headline results (trained on one GAN family — StyleGAN; seed 13 unless noted)
- **RQ1 / H1 — CLIP generalizes better to a *near* unseen generator (SD v1.4):** frozen CLIP (C4)
  **AUROC ≈ 0.92** vs frozen ImageNet ResNet (C1) **≈ 0.84**; 95% bootstrap CIs disjoint (3 seeds).
- **Headline failure — *all four* detectors collapse *below chance* on modern text-to-image faces**
  (SDXL / Flux / DALL·E 3): AUROC **0.17–0.48** — they rate modern fakes as *more real than real*.
  Fine-tuning the backbone (C2) does **not** rescue it; the failure is universal across detector design.
- **RQ2 / H2 — robustness:** every condition degrades under JPEG compression, with a cliff at quality ≤ 10.
- **RQ3 / H3 — calibration:** ECE rises from **≈ 0.02** (in-distribution) to **≈ 0.24** on the unseen
  generator, and in-distribution temperature scaling does **not** repair CLIP's shift-induced
  over-confidence — the core security-reliability finding.

> Directional reproduction: one training GAN family, ~6k images/class subset, single-seed for non-core
> conditions — see the report's *Threats to Validity*. Not a state-of-the-art claim.
