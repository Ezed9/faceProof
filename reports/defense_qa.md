# FaceProof — Defense Q&A (viva prep)

Adversarial cross-questions a reviewer/professor is likely to ask, with honest answers grounded in
our own numbers (`experiments.md`) and the literature. **Rule: never over-claim. Concede real
limitations; separate what we showed from what we infer.** Scope reminder (from `CLAUDE.md`): this is
a *directional reproduction* of Ojha et al. 2023 on faces **plus** a robustness + calibration audit —
not a novel architecture, not an exact reproduction, not a SoTA claim.

Key numbers referenced below (seed 13 unless noted; ~6k/class subset):
- In-dist / SD AUROC: C4 CLIP 0.997 / 0.907, C1 ResNet 0.876 / 0.834, C2 ft-ResNet 0.996 / 0.942, C3 DCT+SVM 0.704 / 0.630.
- H1 (3 seeds): C4 0.920±0.015 vs C1 0.836±0.021; bootstrap 95% CI **CLIP [0.897, 0.915] vs ResNet [0.818, 0.848] — disjoint**.
- Modern T2I AUROC (all below 0.5): C4 SDXL 0.310 / Flux 0.411 / DALL·E3 0.343; C1 0.304 / 0.273 / 0.182; C2 0.484 / 0.322 / 0.170; C3 ≈ 0.35–0.37.
- Calibration (balanced): ECE C4 0.020→0.239 (in-dist→SD), C1 0.162→0.203; in-dist temperature scaling fixes in-dist (C4→0.009) but **not** cross-gen for CLIP (0.239→0.226).

---

## 1. The below-chance headline (our most surprising claim)

**Q1. Is a below-chance AUROC even sensible, or are your labels / score direction just flipped?**
Below-chance AUROC (< 0.5) means the ranking is *systematically inverted* — on average the probe
scores modern-T2I fakes as **more real than real faces** (e.g., CLIP mean P(synthetic): reals 0.031
vs SDXL fakes 0.014). That is different from AUROC ≈ 0.5 (no signal). We rule out a sign/label bug
three ways: (a) the *same* probes score AUROC > 0.9 in the correct direction in-distribution and on
SD v1.4; (b) real and fake go through *identical* preprocessing (224 + JPEG-90), and the DCT leakage
check is near chance; (c) a label/sign bug would flip *every* condition uniformly, but the inversion
is **generator-selective** (SD v1.4 stays ≈0.9 while SDXL/Flux/DALL·E collapse) and varies by
generator (DALL·E3 deepest). Genuine AUROC inversion (not a sign artifact) is documented for
training-free detectors by Zhou & Wang (2026, arXiv:2606.20488), and below-50%-*accuracy* collapse
for supervised detectors by Petrželková & Čech (2024) and Wu et al. (CCS 2024).

**Q2. "Below chance" or just "near chance" dressed up? Give the numbers and CIs.**
Values range 0.17–0.48. We should say "below chance" only where it's clearly separated from 0.5.
The deep ones are unambiguous (C1 DALL·E3 0.182, C2 DALL·E3 0.170). The borderline ones (C2 SDXL
0.484) we should call "at/near chance," not "below chance." **Honesty gap to concede:** the T2I
numbers are single-seed with no bootstrap CI (unlike H1). We report them as a strong directional
finding and explicitly flag that the *most surprising* number is the one without multi-seed
confirmation (see Q13).

**Q3. What actually explains it?**
We *infer* (not prove) a mechanism from adjacent literature: detectors learn generator-specific
"fingerprints," not universal fakeness (Petrželková & Čech 2024). Modern T2I faces are more
prototypical/symmetric/smooth along the very axes a StyleGAN-vs-FFHQ boundary keys on — the
human-perception analog is Miller et al. (2023, *Psychological Science*), where StyleGAN2 faces were
judged human *more often* than real faces; Chandrasegaran et al. (2022) show color statistics are a
transferable forensic feature that could likewise invert. **State clearly this is a hypothesis
assembled from related work, not a mechanism we demonstrated for our setup.**

---

## 2. Reproduction & generalization

**Q4. You trained on one GAN family (StyleGAN). Isn't a single training generator a fatal flaw, not a finding?**
It's a real limitation, and it's *why* the collapse is expected rather than shocking: Park & Owens
(Community Forensics, 2024/25) show generalization scales with training-generator diversity, and both
Ojha 2023 and Petrželková & Čech 2024 use a single base family. Our contribution isn't "we were
surprised"; it's a *face-specific, modern-T2I-specific, magnitude-and-direction* datapoint (below
chance, and unrescued by fine-tuning) that the field suspects but rarely quantifies on 2024-era
generators.

**Q5. Frozen CLIP generalization is old news — MFCLIP and adapter methods already beat it. Why care about C4?**
Agreed, and we say so: MFCLIP (2024/25, arXiv:2409.09724) *adapts* CLIP (image+noise+text) and is
designed for exactly diffusion-face generalization; Forensics-Adapter-style methods do too. Our
frozen-CLIP C4 is deliberately a **floor/reproduction baseline** (project scope forbids fine-tuning
CLIP). The contribution is the *audit lens* (robustness + calibration + below-chance-on-T2I), not a
better detector.

**Q6. Is the H1 margin (CLIP > ResNet) actually significant, given the subset and single training run?**
Yes, for H1 specifically: over 3 seeds the 95% bootstrap CIs are disjoint (CLIP [0.897, 0.915] vs
ResNet [0.818, 0.848]). Caveat: bootstrap-over-images captures sampling noise, and 3 seeds capture
some train-split noise, but neither captures full model-training variance. We call H1 "statistically
supported," not "proven."

---

## 3. Fine-tuning / C2 (the sharpest challenge)

**Q7. You say fine-tuning (C2) doesn't rescue the collapse — but Ricker et al. 2022 show retraining on diffusion gives near-perfect detection. Is your C2 just underpowered?**
Be precise about what C2 is: ResNet-50 fine-tuned **end-to-end on the StyleGAN+real training split
only** — it never sees any T2I data. Ricker et al. (2022, arXiv:2210.14571) is the key counter-point:
retraining *on the target generator* fixes detection (and even generalizes back to GANs), which means
the below-chance collapse is a **training-distribution problem, not an information-theoretic limit** —
the discriminative signal exists; our detectors just weren't trained on it. So our honest claim is
narrow and correct: *fine-tuning on the seen family does not rescue the unseen family.* We are **not**
claiming modern T2I is undetectable — FakeInversion (Cazenavette et al., CVPR 2024) detects DALL·E 3
well. We should state this explicitly so the claim isn't mis-read as "no detector can work."

**Q8. C2 used early stopping on val AUROC — is the C2 number stable?**
C2 early-stops on val AUROC (patience 2). Single training run, so C2 is the least seed-stable
condition; we report it as an extension, and its *direction* (in-dist/SD strong, T2I below chance)
matches the frozen conditions, which is what matters for the universal-collapse claim.

---

## 4. Robustness (H2)

**Q9. Isn't "detectors degrade under JPEG" trivially expected?**
Directionally yes — the value is the *shape*: graceful decline to ~q30 then a cliff at q ≤ 10, with
CLIP more robust than ResNet at every level, and role-inversion under stress (at q10, CLIP cross-gen
≈0.82 > ResNet in-dist ≈0.69). Concede the low-quality tail is single-seed and noisy (the CLIP/SD q10
bump is noise, not a real uptick). Only JPEG is run; resize/blur/noise are coded but not in results.

---

## 5. Calibration (H3)

**Q10. Calibration degrading under shift, and temperature scaling failing OOD — aren't both textbook?**
Yes, both are established generally (Guo et al. 2017; TS is a single global scalar fit in-distribution
and is known not to transfer OOD). Our novelty is the *pairing*: generator-shift × faces × the
biometric-security reading (a detector that stays accurate-looking but becomes confidently wrong is
the actual enrollment threat). We claim the combination and the security framing, **not** the
calibration-degrades phenomenon itself.

**Q11. Your ECE dropped from 0.39 to 0.24 when you "balanced" the groups. Isn't that cherry-picking?**
No — it's a correctness fix we should volunteer. ECE depends on class prevalence; the raw cross-gen
set was ~85% synthetic, which inflates ECE. We recompute on balanced 50/50 groups for a fair in-dist
vs cross-gen comparison (raw 0.39 → balanced 0.24). We report the balanced number and disclose the
balancing. The headline (ECE up ~12× for CLIP, 0.020→0.239, while AUROC stays ~0.9) holds either way.

---

## 6. Methodology & leakage

**Q12. How do you know results aren't a compression/preprocessing artifact?**
Rule #1: real and fake are routed through identical resize + JPEG-90 re-encoding. Rule #2: the
DCT+SVM leakage probe on real vs StyleGAN scored **near chance** — a frequency-only shortcut can't
separate the classes, so compression isn't a label proxy. Splits are source/identity-disjoint and
unseen generators are test-only (Rule #3).

**Q13. The reals are ~256px (FFHQ) and the T2I fakes are ~1024px. Is the below-chance collapse just a resolution mismatch?**
This is the strongest methodological threat and we address it head-on with
`notebooks/ablation_resolution.ipynb`: it re-preprocesses the T2I fakes through a 256px intermediate
(matching the reals' origin) before the standard 224/JPEG-90 pipeline and re-scores the saved probes.
If AUROC stays below chance under matched resolution, the collapse is not a resolution artifact; if it
jumps toward 0.5, resolution asymmetry contributed. **We will report whichever the ablation shows.**
Note the asymmetry was *constant* across Day-3 (SD ≈0.9) and Day-5 (T2I below chance) evals, so it
can't by itself explain why SD is detectable and SDXL isn't — but the ablation makes this rigorous.

**Q14. Which "real" faces are paired with the T2I fakes?**
Held-out FFHQ reals (label 0, `split == test_indist`) — not training images, and the *same* real pool
used for the in-distribution eval. So the T2I comparison is genuinely held-out real vs unseen fake,
not two mismatched real collections.

**Q15. DALL·E 3 has only n=1123 — is that enough?**
It's the smallest T2I set and the noisiest single number; we flag n explicitly and lean on the fact
that all three T2I generators (and all four detectors) collapse, so the finding doesn't hinge on
DALL·E 3 alone.

---

## 7. Citations & honesty

**Q16. You list MFCLIP / "Han et al. 2025" / "OmniFake" as related SoTA — did you verify them?**
- **MFCLIP** — verified, directly on-topic (arXiv:2409.09724); cite as the "adapt CLIP beats frozen CLIP" reference.
- **"Han et al. 2025"** — the closest verified match is a *video* deepfake paper (CVPR 2025, facial-component-guided foundation-model adaptation); only loosely relevant. We should either re-specify the exact paper or cite it as video-domain context, not as a still-face SoTA we're compared to.
- **"OmniFake"** — **could not be verified** under that name; the closest ("Omni-Fake," arXiv:2605.01638) is a different-scope multimodal social-media benchmark. **We remove "OmniFake" from citations** rather than cite something adjacent-sounding.

Being able to say "we checked our own related-work list and dropped an unverifiable citation" is a
strength, not a weakness.

---

## 8. Scope & framing

**Q17. Is the biometric-security framing doing real work, or is it dressing?**
Per scope, the security framing is motivation + the demo, not a built pipeline. The real security
contribution is the reliability finding: a detector that looks good in-distribution (AUROC > 0.9,
ECE ~0.02) can silently and confidently fail on the *next* generator family — exactly the threat model
for synthetic-identity injection at enrollment, where the attacker uses the newest generator, not the
one the detector was trained on.

**Q18. One-sentence honest summary of the contribution?**
"On faces, we reproduce the frozen-CLIP > frozen-CNN generalization result on a near generator, and
then show — with a robustness and calibration audit — that every detector we tried (frozen CLIP/ResNet,
fine-tuned ResNet, DCT frequency) collapses below chance and becomes confidently wrong on modern
text-to-image faces, which is a reliability failure that matters for enrollment screening."

---

## Counter-evidence to be ready for (steelman the other side)
- **Ricker et al. 2022** — retraining on diffusion → near-perfect detection; the collapse is fixable with the right training data (bears on Q7).
- **Cazenavette et al. 2024 (FakeInversion)** — diffusion-inversion features *do* generalize to DALL·E 3; "no method works" would be false.
- **Park & Owens 2024/25 (Community Forensics)** — diversity of training generators fixes generalization; our single-family setup is a known, fixable bottleneck.
- **Chandrasegaran et al. 2022** — transferable forensic (color) features exist; a stronger detector might not invert.

## Weaknesses we volunteer before being asked
Single training GAN family; ~6k/class subset; single-seed for T2I/calibration/C2 (only H1 is multi-seed + CI);
DALL·E 3 n=1123; resolution asymmetry (mitigated by the ablation); only JPEG in the robustness results;
frozen-CLIP is a floor, not SoTA; the mechanism for below-chance inversion is inferred, not demonstrated.
