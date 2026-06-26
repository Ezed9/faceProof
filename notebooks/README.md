# Notebooks (run on free Colab/Kaggle)

Run in this order (mirrors the 2-week plan):

1. **01_preprocess_and_leakage.ipynb** *(Day 1–2)* — download data, align/crop + compression-match,
   then run the DCT leakage check. Gate: leakage accuracy near chance.
2. **02_cache_features.ipynb** *(Day 2–3)* — extract & cache CLIP and ResNet features.
3. **03_train_probe_crossgen.ipynb** *(Day 3–6)* — train probes (C1, C4), evaluate in-distribution
   and cross-generator (AUROC/EER), run robustness + calibration.

Each notebook just imports from `src/` and calls the functions there. Keep heavy training on
Kaggle (P100, 30 GPU-hrs/week); use Colab for quick work. Cache features to Drive once.

> Tip: at the top of each Colab notebook, `!git clone <your repo>` then `%cd` into it and
> `!pip install -r requirements.txt`.
