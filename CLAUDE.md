# CLAUDE.md — FaceProof project context

> Read this first. It is the single source of truth for this project. It is written so a fresh
> Claude Code session has full context without any prior chat history.

## What this project is

**FaceProof** is a **research-internship project**: a *reproduction + reliability audit* of
AI-generated **face** detection, framed as a cybersecurity control (screening synthetic-identity
injection at biometric enrollment).

**Honest scope (do not drift from this):**
- It is a **directional reproduction** of Ojha et al. 2023 (frozen CLIP features generalize across
  image generators better than trained CNNs), applied to **faces**, **plus** a robustness +
  calibration audit.
- It is **NOT** a novel architecture, **NOT** an exact reproduction of any paper's numbers, and
  **NOT** a state-of-the-art claim. Never write marketing/novelty claims into the repo or report.
- The biometric/security framing is **motivation + the demo**, not a built production pipeline.

## Who I'm working with
- Beginner at ML/PyTorch (comfortable with basic Python). Uses AI assistants (you) to write code,
  but must **understand and be able to defend every line** — this is a research internship and they
  will be questioned. Explain things simply; pseudocode before coding.
- Constraints that are non-negotiable: **everything must be 100% FREE** (no paid datasets, APIs, or
  compute), and the plan must stay **completable** and **not over-promise**.

## Research questions & hypotheses (see HYPOTHESES.md)
- **RQ1/H1:** frozen CLIP features generalize to *unseen generators* better than frozen ImageNet
  features (confirmatory).
- **RQ2/H2:** both detectors degrade under JPEG / real-world image corruption.
- **RQ3/H3:** calibration worsens under generator shift — detectors become *confidently wrong* on
  unseen generators (the headline security-reliability finding).

## Detector conditions
| Cond. | Detector | Tier |
|-------|----------|------|
| C1 | Frozen ImageNet ResNet-50 + logistic probe | **core** |
| C4 | Frozen CLIP ViT-L/14 + logistic probe | **core** |
| C2 | Fine-tuned ResNet-50 (end-to-end) | extension |
| C3 | DCT log-spectrum + linear SVM (frequency baseline) | extension |

The two core conditions are just "extract frozen features once → fit logistic regression" — minimal
engineering, no backbone training. **Never fine-tune CLIP** (frozen is the claim + stays within free
GPU limits). Cache features to disk/Drive once.

## Datasets (all FREE — never commit to git; configure in configs/data.yaml)
- **Real faces + StyleGAN fakes (TRAIN):** Kaggle "140k real and fake faces".
- **Unseen test 1 (core):** **SFHQ Part 2** = Stable Diffusion v1.4 faces.
- **Unseen test 2 (cheap optional):** SFHQ Part 4 = Stable Diffusion v2.1 faces.
- **Multi-generator extension:** **SFHQ-T2I** = SDXL / Flux / DALL-E 3 faces (labelled in CSV).
- **Do NOT use:** CIFAKE (it's CIFAR objects, not faces). SFHQ **Part 3** (StyleGAN2) and Part 1 are
  skipped (redundant with the StyleGAN training data; GAN→GAN isn't the research question).

`configs/data.yaml`: set each source's `dir` to the folder that directly contains that group's
images (`label: 0`=real, `label: 1`=fake). On Kaggle use `/kaggle/input/...` (mounted read-only);
on Colab use absolute `/content/...` paths. Verify with `src/data.build_manifest`.

## Repo structure & file purposes
```
faceproof/
├── CLAUDE.md            # this file
├── README.md            # public overview
├── HYPOTHESES.md        # RQs, hypotheses, contributions, scope
├── configs/             # data.yaml, model.yaml, experiment.yaml (NO hard-coded settings in code)
├── src/
│   ├── preprocessing.py # Day 1: face align/crop + JPEG compression matching (MTCNN→center-crop fallback)
│   ├── leakage_check.py # Day 2: DCT+SVM shortcut test (must be near chance before trusting results)
│   ├── features.py      # Days 2-3: cache CLIP (open_clip) & ResNet (torchvision) features
│   ├── probe.py         # Days 3-6: logistic-regression probe on cached features
│   ├── metrics.py       # AUROC, EER, ECE, bootstrap CI
│   ├── corruptions.py   # Day 4: JPEG/resize/blur/noise for the robustness sweep
│   └── data.py          # manifest + leakage-safe (source/identity-disjoint) splits
├── notebooks/           # Colab notebooks (run order in notebooks/README.md)
├── data/  models/       # gitignored (datasets, feature caches, checkpoints)
├── reports/             # outline.md + figures + final technical report
├── demo/                # Gradio demo for Hugging Face Spaces (Week 2)
└── experiments.md       # RUNNING LOG — update every run (date, config, seed, metric, value)
```

## Non-negotiable methodology rules
1. **Compression/resolution matching:** route real and fake through identical resize + JPEG
   re-encoding, or the model cheats on file artifacts. (preprocessing.py handles this.)
2. **Leakage check first:** run leakage_check.py on ~500/class; if DCT+SVM separates real/fake well
   (>~0.80), preprocessing is leaking — fix before any model claim.
3. **Leakage-safe splits:** no source/identity crosses train/val/test; unseen generators are
   test-only. (data.py.)
4. **Frozen CLIP only** for the core; cache features; never fine-tune CLIP.
5. **Honest reporting:** directional (not exact) reproduction; report only what's actually done;
   single-seed results stated honestly; multi-seed + bootstrap CIs where feasible.
6. **Log every run** in experiments.md.

## The 2-week intensive plan (full-time + AI). One gate per week-day; cut from the bottom if behind.
**Week 1 (core results):**
- Day 1 — setup + data + preprocessing (compression matching) + hypotheses. *Gate: matched real/fake batch loads.*
- Day 2 — leakage check + cache CLIP features + C4 probe in-distribution. *Gate: leakage ~chance AND in-dist AUROC > ~0.9.*
- Day 3 — C1 (frozen ResNet probe) + first cross-generator (StyleGAN→SD). *Gate: generalization gap table exists (RQ1).*
- Day 4 — robustness: JPEG sweep (then resize/blur/noise). *Gate: JPEG curve for C1,C4 (RQ2).*
- Day 5 — multi-generator via SFHQ-T2I (SDXL/Flux/DALL-E). *Gate: ≥2 unseen generators.*
- Day 6 — calibration: ECE + reliability + temperature scaling. *Gate: AUROC|EER|ECE table (RQ3).*
- Day 7 — buffer + 3 seeds + bootstrap CIs. *Gate: core results locked with uncertainty.*

**Week 2 (extensions, artifact, writeup):**
- Day 8 — C2 fine-tuned ResNet + C3 frequency baseline. *Gate: all 4 conditions in table.*
- Day 9 — finalize all numbers (freeze experiments).
- Day 10 — figures (system diagram, cross-gen table, robustness curves, reliability diagrams).
- Day 11 — Gradio demo (HF Spaces) + repo polish + reproducibility pass.
- Day 12 — report part 1 (abstract, intro, contributions, related work, method).
- Day 13 — report part 2 (experiments, results, threats to validity, limitations, conclusion).
- Day 14 — polish + slides + backup demo recording + buffer.

**Cut order if behind (keep core + report; never cut leakage controls or honesty):**
attack-then-defend mock → full degradation (keep JPEG) → C3 → C2 → 3rd generator → multi-seed/CIs.

## Current status
**Day 1 — COMPLETE (2026-06-26).** Done: repo scaffolded + pushed to GitHub; 140k real+fake +
SFHQ Part 2 (SD v1.4) downloaded to Drive; compression-matched preprocessing run (224×224,
JPEG-90) via `src/preprocessing.py`; manifest + leakage-safe splits built (`src/data.py`);
unseen generator (SD) confirmed test-only; Day 1 gate passed (matched real/fake batch loads).
**Next actions (Day 2):** (1) run DCT leakage check (`src/leakage_check.py`) on ~500/class from
crops — must be near chance before trusting any result; (2) extract + cache CLIP ViT-L/14
features (`src/features.py`); (3) fit logistic probe on in-distribution data (Day 2 gate: in-dist
AUROC > ~0.9).

## Key reference papers
- Ojha, Li, Lee 2023 — *Towards Universal Fake Image Detectors...* (the anchor; arXiv 2302.10174).
- Frank et al. 2020 — frequency analysis of fakes (basis for C3).
- Radford et al. 2021 — CLIP. Guo et al. 2017 — calibration / temperature scaling.
- Current SoTA to cite as related (do NOT claim to beat): MFCLIP 2024/25, Han et al. 2025, OmniFake.

## Other project docs (in the PARENT folder ../, reference only)
- `../Research_Proposal_FINAL_verified_free.pdf` — the authoritative proposal.
- `../Execution_Plan_2weeks.md` — the day-by-day plan (source of the schedule above).
- `../Reading_Notes_Ojha2023_explained.md` — beginner explanation of the anchor paper.

## Working conventions for Claude Code
- Keep everything **config-driven** (read from configs/, no hard-coded paths/params in src).
- Prefer small, readable functions; add comments a beginner can follow; **pseudocode before coding**.
- After writing code, explain what it does and how to run it; remind the user to log the run.
- Stay within the honest scope above; if asked to add novelty/SoTA claims, push back.
