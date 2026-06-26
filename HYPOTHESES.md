# Research Questions, Hypotheses, and Scope

## Research questions
- **RQ1.** On faces, does a frozen-CLIP probe generalize to an *unseen* generator better than a frozen-ImageNet-CNN probe?
- **RQ2.** How do the detectors degrade under real-world image corruption (JPEG, resize, blur, noise)?
- **RQ3.** Does confidence calibration worsen under generator shift?

## Hypotheses
- **H1 (confirmatory).** We expect to *confirm* that frozen CLIP features generalize to unseen
  generators better than frozen ImageNet features under generator shift (cf. Ojha et al. 2023),
  now in the faces domain.
- **H2.** Detection performance (AUROC) degrades for *both* representations as JPEG compression increases.
- **H3.** Calibration error (ECE) increases under generator shift — detectors become *confidently wrong*
  on unseen generators, even when ranking (AUROC) is still reasonable.

## Contributions (planned)
1. A directional reproduction of CLIP-based synthetic-face detection on faces.
2. A cross-generator generalization evaluation across multiple unseen generators.
3. A robustness audit under real-world image degradation.
4. A calibration study under generator shift (the security-reliability finding).
5. A reproducible, free, open-source implementation + demo.

## Scope boundaries
- Screens *digital synthetic-media injection* at enrollment. Not liveness, not hardware/camera
  integrity, not adaptive white-box attackers.
- Directional reproduction, not exact reproduction of any paper's metrics.
- Not a novel architecture and not a state-of-the-art claim.
