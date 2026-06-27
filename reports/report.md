# FaceProof — Reproduction + Reliability Audit of AI-Generated Face Detection

> **Draft skeleton (Days 12–13).** Fill bracketed `[…]` from frozen `results.csv` / `reports/table_*.md`.
> Do **not** invent numbers. State single-seed results honestly; add CIs where computed (nb 07).
> Scope per `CLAUDE.md`: directional reproduction of Ojha 2023 on faces + robustness/calibration audit.
> NOT a novel architecture, NOT exact reproduction, NOT a SoTA claim.

## 1. Abstract
We audit frozen-feature detectors of AI-generated **faces** as a biometric security control
(screening synthetic-identity injection at enrollment). Reproducing Ojha et al. (2023) directionally,
we compare a frozen CLIP probe (C4) and a frozen ImageNet ResNet probe (C1) — plus a fine-tuned
ResNet (C2) and a DCT+SVM frequency baseline (C3) — across an unseen-generator shift, image
corruption, and calibration. Key finding: [one-sentence headline, e.g. detectors stay accurate but
become *confidently wrong* under generator shift — ECE rises from [x] to [y]].

## 2. Introduction
Synthetic-identity injection — enrolling a generated face as a real person — is a concrete biometric
threat. Detection accuracy alone is insufficient for a security control: a detector that is
over-confident on inputs unlike its training data fails silently. We therefore measure not just
AUROC/EER but **calibration** under distribution shift.

## 3. Contributions
1. A directional reproduction of the frozen-feature generalization result, specialized to faces.
2. A robustness audit (JPEG and other corruptions) of both core detectors.
3. A calibration audit showing [over-confidence / ECE behaviour] under unseen-generator shift.
(See `HYPOTHESES.md` for RQ/H mapping.)

## 4. Related work
Ojha, Li & Lee (2023) frozen-CLIP universal detectors; Frank et al. (2020) frequency analysis (basis
for C3); Radford et al. (2021) CLIP; Guo et al. (2017) temperature scaling. Recent detectors we cite
but do **not** claim to beat: MFCLIP (2024/25), Han et al. (2025), OmniFake.

## 5. Method
- **Conditions:** C1 frozen ResNet-50 + logistic probe; C4 frozen CLIP ViT-L/14 + logistic probe;
  C2 fine-tuned ResNet-50; C3 DCT log-spectrum + linear SVM.
- **Datasets:** train = 140k real (FFHQ) + StyleGAN fakes; unseen test = SFHQ Part 2 (SD v1.4) and
  SFHQ-T2I (SDXL/Flux/DALL-E). All free; never committed.
- **Preprocessing + compression matching:** all images aligned/cropped to 224×224 and re-encoded at
  JPEG-90 so compression cannot become a label proxy.
- **Leakage check:** DCT+SVM on crops scored [acc] (≈ chance) → no shortcut. [Day 2]
- **Splits:** source/identity-disjoint; unseen generators are test-only.

## 6. Experiments
In-distribution (held-out StyleGAN); cross-generator (leave-one-generator-out to SD / SDXL / Flux /
DALL-E); robustness sweep (JPEG core; resize/blur/noise extension); calibration (ECE + reliability +
temperature scaling). Seeds [13, 37, 71]; bootstrap CIs over images.

## 7. Results
- **Generalization gap (AUROC):** see `reports/table_auroc.md`. C1 in-dist [..] → SD [..] (gap [..]);
  C4 in-dist [..] → SD [..] (gap [..]). [H1: CLIP gap </> ResNet gap.]
- **Robustness:** `reports/figures/robustness_jpeg.png`. AUROC at JPEG-90→10: C4 [..]→[..], C1 [..]→[..].
- **Calibration:** `reports/table_calibration.md`. ECE in-dist vs unseen, before/after temperature.
- **Uncertainty:** cross-gen AUROC CIs [..] (nb 07).

## 8. Threats to validity
Internal: residual compression/resolution leakage (mitigated + checked). External: faces only, limited
generators, single training GAN family. Construct: AUROC ≠ deployment risk; EER threshold choice.

## 9. Limitations & ethics
Screening, not liveness; no adaptive/adversarial attacker; dual-use (the same detector informs
evasion). Datasets are synthetic or public faces; no new personal data collected.

## 10. Conclusion
[Restate the calibration/over-confidence finding and its security implication; future work: adaptive
attacker, more generators, on-device calibration.]
