# FaceProof — Reproduction + Reliability Audit of AI-Generated Face Detection

*A directional reproduction of Ojha et al. (2023) on faces, plus a robustness and calibration audit,
framed as a biometric enrollment-screening control.*

> **Scope (honest).** This is a **directional reproduction** of the frozen-feature generalization
> result of Ojha et al. (2023), specialized to **faces**, **plus** a reliability audit (robustness +
> calibration). It is **not** a novel architecture, **not** an exact reproduction of any paper's
> numbers, and **not** a state-of-the-art claim. Numbers are single-seed on a ~6k/class subset unless
> a seed set is stated; only the core H1 result carries multi-seed + bootstrap CIs. All figures and
> tables trace to `../experiments.md` and `reports/results.csv`.

---

## 1. Abstract

We audit frozen-feature detectors of AI-generated **faces** as a biometric security control
(screening synthetic-identity injection at enrollment). Reproducing Ojha et al. (2023)
directionally, a frozen **CLIP** ViT-L/14 probe (C4) generalizes to an unseen *near* generator
(Stable Diffusion v1.4) better than a frozen ImageNet **ResNet-50** probe (C1): AUROC **0.92 vs 0.84**,
stable across three seeds, with non-overlapping 95% bootstrap CIs (computed within the seed-13
evaluation). The headline finding is a
reliability failure: on **modern text-to-image** generators (SDXL, Flux, DALL·E 3), **all four**
detectors we test — frozen CLIP (C4), frozen ResNet (C1), a fine-tuned ResNet (C2), and a DCT
frequency baseline (C3) — collapse to **below-chance AUROC (0.17–0.48)**, systematically scoring
modern fakes as *more real than real* faces; end-to-end fine-tuning does not rescue this. Detection
also degrades under JPEG compression (a cliff at quality ≤ 10), and **calibration** worsens sharply
under generator shift: expected calibration error (ECE) for CLIP rises from **0.020** in-distribution
to **0.239** on the unseen generator while AUROC stays ≈ 0.9, and in-distribution temperature scaling
does not repair it. For a security control, a detector that stays accurate-looking but becomes
*confidently wrong* on the next generation of generators is the operative risk.

## 2. Introduction

Synthetic-identity injection — enrolling a generated face as if it were a real person — is a concrete
biometric threat: an attacker submits an AI-generated face at enrollment to create a credentialed
identity that never existed. A natural defense is an AI-face detector at the enrollment gate. But
**accuracy alone is insufficient** for a security control. Two failure modes matter more than average
accuracy: (i) **generalization** — the attacker uses whatever generator is newest, not the one the
detector was trained on; and (ii) **calibration** — a detector that is over-confident on inputs unlike
its training data fails *silently*, passing a fake with high "real" confidence and giving an operator
no signal to review. We therefore measure not just AUROC/EER but **robustness** to image corruption
and **calibration** under distribution shift.

## 3. Contributions

1. A **directional reproduction** of the frozen-feature generalization result (frozen CLIP > frozen
   ImageNet CNN on an unseen generator), specialized to faces and reported with multi-seed bootstrap CIs.
2. A **robustness audit** (JPEG sweep) of the core detectors, showing graceful degradation then a cliff.
3. A **calibration audit** (ECE, reliability diagrams, temperature scaling) showing over-confidence
   emerges under generator shift and resists in-distribution temperature scaling — the headline
   security-reliability finding.
4. A **multi-generator stress test** on 2024-era text-to-image faces (SDXL/Flux/DALL·E 3) showing a
   **universal below-chance collapse** across four detector designs, unrescued by fine-tuning.

(RQ/H mapping in `../HYPOTHESES.md`.)

## 4. Related work

**Anchor.** Ojha, Li & Lee (2023) show that training a CNN end-to-end for real-vs-fake classification
generalizes poorly to unseen generators, because the network becomes asymmetrically tuned to the
training generator's fakes; classifying in a *frozen* CLIP embedding space (linear probe or
nearest-neighbor) generalizes far better. Trained only on ProGAN (objects, not faces), their CLIP
probe reaches ~82–84% average accuracy on unseen diffusion/autoregressive generators versus ~53–58%
(near chance) for a CNN baseline. FaceProof reproduces this specific comparison (frozen CLIP probe vs
a frozen-CNN probe) in the **faces** domain, swapping ProGAN→StyleGAN for training and their object
generators for face-specific diffusion/T2I generators.

**Frequency baselines.** Frank et al. (2020) show GAN upsampling leaves periodic spectral artifacts
that a simple DCT+SVM can exploit — the basis for our C3. The literature is clear this does **not**
transfer to diffusion: Ricker et al. (2022) find diffusion models do not reproduce the GAN-style
high-frequency signature (they *underestimate* high-frequency density), so GAN-trained frequency
detectors fail on diffusion without retraining — which predicts our C3 result.

**Modern detectors (cited, not claimed to beat).** Post-Ojha work *adapts* rather than freezes CLIP:
MFCLIP (2024/25) is a multi-modal fine-grained CLIP adaptation targeting exactly diffusion-face
generalization; Han et al. (2025, CVPR) adapt a foundation model for (video) deepfake detection via
facial-component guidance. Our frozen-CLIP C4 is deliberately a **floor/reproduction baseline**, not a
competitor to these.

**Corroboration for the below-chance collapse.** Detectors failing — often below the 50% line — on
newer generators is increasingly documented. Petrželková & Čech (2024) report face detectors dropping
to ≈ chance on unseen generators (a GAN-trained method at 48.3% accuracy on an SD-based face
generator); Wu et al. (2024, ACM CCS) report 23.7–41.0% accuracy on DALL·E 2; Zhou & Wang (2026)
document genuine AUROC *inversion* (< 0.5, not just near-chance) for training-free detectors. Our
result is consistent with, and corroborated by, this adjacent literature; to our knowledge the exact
combination (Ojha-style frozen probes, on faces, on SDXL/Flux/DALL·E 3, reported as AUROC) has not
been published, so we frame it as corroborated rather than identical to a prior result.

**Counter-evidence (steelman).** The collapse is a *training-distribution* problem, not an
information-theoretic limit: Ricker et al. (2022) show retraining on diffusion restores near-perfect
detection (and generalizes back to GANs); FakeInversion (Cazenavette et al., 2024) detects DALL·E 3
well using diffusion-inversion features; Community Forensics (Park & Owens, 2024/25) shows
generalization scales with training-generator diversity. We therefore do **not** claim modern T2I is
undetectable — only that the specific StyleGAN-trained detectors we test collapse on it.

**Calibration.** Modern networks are over-confident (Guo et al., 2017), and temperature scaling — a
single scalar fit in-distribution — is known not to transfer under distribution shift. We apply this
lens to generator-shift in face detection; the novelty is the *pairing* (shift × faces × security
framing), not the phenomenon.

## 5. Method

**Detector conditions.**

| Cond. | Detector | Tier |
|-------|----------|------|
| C1 | Frozen ImageNet ResNet-50 (2048-d) + logistic-regression probe | core |
| C4 | Frozen CLIP ViT-L/14 (768-d, OpenAI) + logistic-regression probe | core |
| C2 | ResNet-50 fine-tuned end-to-end (StyleGAN+real training split only) | extension |
| C3 | DCT log-spectrum + linear SVM (frequency baseline) | extension |

The two core conditions extract frozen features once and fit a logistic regression — no backbone
training. CLIP is **never fine-tuned** (frozen is the claim). Features are cached to disk.

**Datasets (all free; never committed).** Train: "140k real and fake faces" (FFHQ real + StyleGAN
fakes). Unseen test — near: SFHQ Part 2 (Stable Diffusion v1.4). Unseen test — modern T2I: SFHQ-T2I
(SDXL, Flux, DALL·E 3; labelled by `model_used`). GAN→GAN redundancy (SFHQ Parts 1/3) and non-face
sets (CIFAKE) are excluded by design.

**Preprocessing + compression matching.** Every image — real and fake — is routed through identical
alignment (MTCNN with a center-crop fallback), resized to 224×224, and re-encoded at **JPEG-90**, so
compression/resolution cannot become a label proxy.

**Leakage check.** Before trusting any result we ran a DCT log-spectrum + SVM probe on real vs
StyleGAN crops (~500/class). It scored **near chance** — a frequency-only shortcut cannot separate the
classes — confirming compression matching held and downstream results are not a preprocessing
artifact.

**Splits.** Source/identity-disjoint; unseen generators are **test-only**. Manifest and splits are
built at seed 13 so they are identical across notebooks.

## 6. Experiments

We evaluate: (i) **in-distribution** (held-out StyleGAN); (ii) **cross-generator** to the unseen SD
v1.4 and to each modern T2I generator, each paired with held-out reals so AUROC is well-defined; (iii)
a **robustness** sweep (JPEG quality 90→10 on test images, re-extracting features); and (iv)
**calibration** (ECE with 15 bins, reliability diagrams, and temperature scaling fit on a held-out
validation set). The core H1 comparison is repeated over seeds [13, 37, 71] with percentile bootstrap
CIs (2000 resamples over images). EER is logged alongside AUROC in `results.csv`.

## 7. Results

### 7.1 Cross-generator generalization (RQ1 / H1)

The primary metric is **AUROC on the unseen generator** (higher = generalizes better). On SD v1.4,
frozen CLIP beats frozen ResNet, and the margin is statistically supported (three seeds; disjoint 95%
bootstrap CIs):

| Detector | In-dist AUROC (3 seeds) | Cross-gen SD AUROC (3 seeds) | 95% bootstrap CI (SD) |
|----------|:---:|:---:|:---:|
| C4 CLIP  | 0.996 ± 0.001 | **0.920 ± 0.015** | [0.897, 0.915] |
| C1 ResNet| 0.872 ± 0.004 | 0.836 ± 0.021 | [0.818, 0.848] |

The CIs do not overlap (CLIP lower bound 0.897 > ResNet upper bound 0.848), so **H1 is supported**.
Note these are two *complementary* uncertainty analyses, not one: the ±std quantifies seed-to-seed
training variance (3 re-splits/retrains), while the bootstrap CI quantifies evaluation-sampling
variance over images *within the seed-13 run* (2000 resamples) — the CI is not computed across seeds.
We report cross-gen AUROC as the primary metric rather than the in-dist−cross-gen *gap*: the gap conflates
baseline in-distribution strength (a weak baseline shows a small gap and looks deceptively robust).
See `reports/figures/crossgen_auroc.png`.

### 7.2 Master table: four detectors × five generators

Single-seed (seed 13), full-set AUROC. (Hence small differences from the 3-seed means in §7.1 — e.g.
C1 in-dist reads 0.876 here (seed-13 run) vs 0.872 ± 0.004 there (3-seed mean); same quantity, two
measurements.)

| Condition | In-dist (StyleGAN) | SD v1.4 (near) | SDXL | Flux | DALL·E 3 |
|-----------|:---:|:---:|:---:|:---:|:---:|
| **C4** CLIP (frozen)      | 0.997 | 0.907 | 0.310 | 0.411 | 0.343 |
| **C1** ResNet (frozen)    | 0.876 | 0.834 | 0.304 | 0.273 | 0.182 |
| **C2** ResNet (fine-tuned)| 0.996 | 0.942 | 0.484 | 0.322 | 0.170 |
| **C3** DCT + SVM          | 0.704 | 0.630 | 0.35–0.37 † | 0.35–0.37 † | 0.35–0.37 † |

† C3 on the three T2I generators was logged as a single range (0.35–0.37), not per-generator.
C1/SD is 0.834 in the frozen Day-9 aggregate; the Day-3 log entry rounds the same run to 0.833.
Chance = 0.50. (Master markdown table also at `reports/table_auroc.md`.)

**Reading the table.** Two regimes:

- **Near shift is fine.** In-distribution and SD v1.4 are strong. Notably C2 (fine-tuned) *generalizes
  to SD as well as or better than* frozen CLIP (0.942 vs 0.907) — the "fine-tuning always hurts
  generalization" intuition does **not** hold on the moderate StyleGAN→SD (2022 diffusion) shift.
- **Modern T2I collapses below chance — universally.** Every detector, on every 2024-era T2I generator,
  falls below 0.50 (range 0.17–0.48). The probes rate modern fakes as *more real than real*: CLIP mean
  P(synthetic) is 0.031 on reals vs 0.014 on SDXL fakes; C2's mean P(synthetic) on DALL·E 3 fakes is
  ≈ 0.000 (it is ~100% sure they are real). Because the *same* probes, reals, and preprocessing yield
  0.907 on SD v1.4, the collapse is **generator-driven** (2022 diffusion detectable; 2024 T2I not),
  and end-to-end fine-tuning (C2) does not rescue it. The failure is **universal across detector
  design** (frozen CLIP/ResNet, fine-tuned ResNet, DCT frequency). We call values clearly separated
  from 0.5 "below chance" (e.g., 0.170–0.343); borderline values (C2 SDXL 0.484) are "at chance."

### 7.3 Robustness (RQ2 / H2)

All conditions degrade as JPEG quality drops (`reports/figures/robustness_jpeg.png`): a graceful
decline to ~q30, then a **cliff at q ≤ 10**. CLIP is more robust than ResNet at every quality level;
at q = 10 the roles invert (CLIP cross-gen ≈ 0.82 exceeds ResNet in-distribution ≈ 0.69, and ResNet on
SD falls to ≈ 0.64, a near-failure regime). The low-quality tail is single-seed and noisy (a small
CLIP/SD bump at q10 is noise, not a real effect). Only JPEG is reported; resize/blur/noise are
implemented but not run.

### 7.4 Calibration (RQ3 / H3)

On balanced 50/50 groups (raw cross-gen prevalence was ~85% synthetic, which inflates ECE; we
recompute balanced for a fair comparison):

| Detector | ECE in-dist (raw → temp) | ECE unseen SD (raw → temp) |
|----------|:---:|:---:|
| C4 CLIP   | 0.020 → 0.009 | 0.239 → **0.226** (temperature fails) |
| C1 ResNet | 0.162 → 0.047 | 0.203 → 0.053 (T ≈ 7.9) |

CLIP is near-perfectly calibrated in-distribution but ~12× worse on the unseen generator **while AUROC
stays ≈ 0.9** — it becomes *confidently wrong*, not merely inaccurate. Reliability diagrams
(`reports/figures/reliability.png`) show confident "real" calls on SD fakes (predicted ≈ 0 →
~27% actually synthetic) vs a near-diagonal in-distribution. Crucially, temperature scaling fit
in-distribution **fixes in-distribution ECE but not the shift-induced miscalibration for CLIP**
(0.239 → 0.226). This is the security-reliability headline: the calibration that makes the detector
trustworthy in the lab does not survive the generator shift an attacker exploits. (Balanced ECE also
at `reports/table_calibration.md`.)

### 7.5 System overview

`reports/figures/system_diagram.png` shows the pipeline: compression-matched preprocessing → frozen
feature extraction (CLIP / ResNet) → logistic probe → AUROC / EER / ECE across generators.

## 8. Threats to validity

- **Resolution asymmetry (internal) — tested; this one artifact ruled out.** Reals originate at ~256px (FFHQ) and
  T2I fakes at ~1024px, so the classes reach the 224px crop through different downscaling. We test this
  directly with `notebooks/ablation_resolution.ipynb`, which re-preprocesses T2I fakes through a 256px
  intermediate (matching the reals' origin) and re-scores the saved seed-13 probes. **The below-chance
  collapse survives the control**: matched AUROC stays far below 0.5 on every generator (C4 CLIP: SDXL
  0.310→0.289, Flux 0.411→0.379, DALL·E 3 0.343→0.335; C1 ResNet: 0.304→0.310, 0.273→0.283,
  0.182→0.192) — if anything CLIP drops slightly under matching. So resolution asymmetry is **not** the
  cause; the collapse is generator-driven. This is consistent with the asymmetry being *constant*
  across the SD (≈0.9) and T2I (below-chance) evaluations. The ablation controls for *this one*
  artifact — it does not exclude every non-generator explanation (see the next threat).
- **Content-distribution confound (internal/external) — disclosed, untested.** FFHQ reals are
  photographic, incidentally-posed Flickr faces; SFHQ-T2I fakes are *prompt-generated* and plausibly
  skew toward idealized, well-lit, prototypical compositions (and possibly a narrower demographic
  range). A detector keying on content statistics correlated with the StyleGAN-vs-FFHQ boundary could
  produce the below-chance inversion without any generator-fingerprint story — indeed, our own
  mechanism hypothesis (§7.2; Miller et al. 2023) effectively names this pathway. We did **not** test
  it (a content-matched subset evaluation would be needed). Two things survive the concession: the
  same probes score ≈0.91 on SD v1.4 through the identical pipeline, so something generator-linked
  changed between 2022- and 2024-era models; and the *security* conclusion is unaffected either way —
  whatever the cause, the control confidently passes exactly the images an attacker would submit.
  Only the mechanistic interpretation narrows.
- **Single-seed for the surprising claim.** The below-chance T2I numbers, the calibration numbers, and
  C2 are single-seed; only H1 carries multi-seed + bootstrap CIs. The most surprising result is the
  one without multi-seed confirmation — stated plainly.
- **Subset size / n.** Core evaluations use a ~6k/class subset; DALL·E 3 has n = 1123 (the noisiest
  cell). The universal collapse across three generators and four detectors does not hinge on any single
  cell.
- **Leakage / compression (internal).** Mitigated by identical preprocessing and confirmed by a
  near-chance DCT+SVM leakage probe; residual leakage cannot be fully excluded.
- **C2 instability.** C2 uses early stopping on validation AUROC over a single run; it is the least
  seed-stable condition and is reported as an extension.
- **ECE and prevalence (construct).** ECE depends on class balance; we disclose the balancing (raw
  cross-gen ECE 0.39 → balanced 0.24) and report balanced numbers throughout.
- **Construct validity.** AUROC/ECE are not deployment risk; the EER operating threshold is a policy
  choice. The mechanism for below-chance *inversion* is inferred from adjacent literature, not
  demonstrated here.
- **External validity.** Faces only; a single training GAN family (StyleGAN); a limited set of
  generators. Frozen CLIP is a floor, not the current state of the art (cf. MFCLIP).

## 9. Limitations & ethics

FaceProof is a **screening** control, not a liveness check, and assumes **no adaptive/adversarial
attacker** (an attacker aware of the detector could evade it). The work is dual-use: the same detector
that screens fakes also reveals what a generator must fix to pass. All data is synthetic or public
research faces (FFHQ is CC BY-NC-SA, research-use only); no new personal data was collected. The
biometric framing is motivation and a demo, not a deployed pipeline.

## 10. Conclusion

Reproducing Ojha et al. (2023) on faces, frozen CLIP features generalize to a near unseen generator
better than frozen ImageNet features (0.92 vs 0.84, disjoint CIs). But the reliability audit is the
substance: on modern text-to-image faces, every detector we tried collapses **below chance** and
becomes **confidently wrong**, and end-to-end fine-tuning does not rescue it; calibration that holds
in the lab fails under the generator shift an attacker would exploit, and in-distribution temperature
scaling does not repair it. For enrollment screening, the lesson is that in-distribution accuracy and
calibration are *not* evidence of safety against the next generator. Future work, consistent with the
counter-evidence: train on diverse generators (Community Forensics), evaluate adaptation-based
detectors (MFCLIP, FakeInversion), add an adaptive attacker, and study on-device calibration under
shift. The honest headline is a caution, not a detector: **measure how confidently wrong a face
detector is on unseen generators, because that is where a biometric control silently fails.**

## References

- Ojha, Li & Lee (2023). *Towards Universal Fake Image Detectors that Generalize Across Generative Models.* CVPR. arXiv:2302.10174.
- Frank et al. (2020). *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML. arXiv:2003.08685.
- Ricker, Damm, Holz & Fischer (2022). *Towards the Detection of Diffusion Model Deepfakes.* arXiv:2210.14571.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). ICML.
- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
- MFCLIP (2024/25). *Multi-modal Fine-grained CLIP for Generalizable Diffusion Face Forgery Detection.* arXiv:2409.09724.
- Han et al. (2025). *Towards More General Video-based Deepfake Detection through Facial Component Guided Adaptation for Foundation Model.* CVPR. *(video domain — related context.)*
- Petrželková & Čech (2024). *Detection of Synthetic Face Images: Accuracy, Robustness, Generalization.* arXiv:2406.17547.
- Wu, Shen, Backes & Zhang (2024). *Image-Perfect Imperfections.* ACM CCS. arXiv:2408.17285.
- Zhou & Wang (2026). *How Fragile Are Training-Free AI-Generated Image Detectors?* arXiv:2606.20488.
- Cazenavette, Sud, Leung & Usman (2024). *FakeInversion.* CVPR. arXiv:2406.08603.
- Park & Owens (2024/25). *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors.* arXiv:2411.04125.
- Chandrasegaran et al. (2022). *Discovering Transferable Forensic Features.* ECCV. arXiv:2208.11342.
- Miller et al. (2023). *AI Hyperrealism: Why AI Faces Are Perceived as More Real Than Human Ones.* Psychological Science 34(12).
- Beniaguev (2022). *SFHQ / SFHQ-T2I: Synthetic Faces High Quality dataset.* github.com/SelfishGene.
