# Technical Report — Outline

Target length: 4–8 pages. Write incrementally from Day 4 onward; assemble in Week 2.

1. **Abstract** — problem, what we did (directional reproduction + reliability audit), key finding.
2. **Introduction** — synthetic-identity injection as a security threat; why reliability (not just accuracy) matters.
3. **Contributions** — numbered (see HYPOTHESES.md).
4. **Related work** — Ojha 2023, Frank 2020, CLIP, MFCLIP 2024/25, Han et al. 2025, robustness benchmarks.
5. **Method** — four conditions (C1–C4); datasets; preprocessing + compression matching; leakage check.
6. **Experiments** — in-distribution; cross-generator (LOGO); robustness sweep; calibration.
7. **Results** — per-generator AUROC/EER/ECE table; robustness curves; reliability diagrams.
8. **Threats to validity** — internal (leakage/compression), external (faces-only, limited generators),
   construct (AUROC vs deployment risk).
9. **Limitations & ethics** — screening not liveness; no adaptive attacker; dual-use.
10. **Conclusion** — the calibration/over-confidence finding; future work.

Figures: system diagram, cross-generator table, robustness curves, reliability diagrams.
