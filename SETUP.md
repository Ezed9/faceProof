# SETUP — reproducing FaceProof (100% free)

This is the runnable-from-scratch guide the README points to. Everything here uses **free** compute
(Google Colab T4 or Kaggle P100) and **free** datasets. Nothing large is committed to git — datasets,
feature caches, probes, and results live on Google Drive and are rebuilt by the notebooks.

> TL;DR: get a Kaggle API token → run `notebooks/01…10` in order on Colab with Drive mounted →
> everything lands under `MyDrive/faceproof/`. Numbers are already frozen in `experiments.md`; you
> only need this to re-derive them.

## 1. Compute

- **Colab (recommended):** free T4 GPU. Notebooks mount Drive at `/content/drive/MyDrive/faceproof`.
- **Kaggle:** free P100 (30 GPU-hrs/week); datasets mount read-only under `/kaggle/input/...`.
- Notebooks **02 / 04 / 05 / 08** and the ablation need a **GPU**; **03 / 06 / 07 / 09 / 10** are
  CPU-light (they reuse cached features).

## 2. Datasets (all free — never commit them)

| Role | Dataset | Generator | Source |
|------|---------|-----------|--------|
| Real + train fakes | "140k real and fake faces" | FFHQ real + StyleGAN | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| Unseen test 1 (core) | SFHQ **Part 2** | Stable Diffusion v1.4 | https://github.com/SelfishGene/SFHQ-dataset |
| Multi-generator (extension) | SFHQ-**T2I** | SDXL / Flux / DALL·E 3 | https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models |

Do **not** use CIFAKE (objects, not faces) or SFHQ Part 1/3 (redundant GAN faces). Respect licenses
(FFHQ is CC BY-NC-SA — research use only).

### Kaggle API token (needed to download inside Colab/Kaggle)
1. kaggle.com → Account → **Create New API Token** → downloads `kaggle.json`.
2. In a Colab cell: upload `kaggle.json`, then
   ```bash
   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models -p /content/t2i --unzip
   ```
   (nb08 and the ablation notebook do this for you and skip the download if the CSV is already present.)

## 3. Config-driven paths

All settings live in `configs/` — **no paths or params are hard-coded in `src/`**. Point each source
to the folder that directly contains that group's images (`label: 0` = real, `label: 1` = fake).

- `configs/data.yaml` — `image_size: 224`, `jpeg_match_quality: 90`, the `sources:` folders, and the
  leakage-safe split (`train_generators: [stylegan]`, `test_unseen: [sd]`, `seed: 13`).
- `configs/model.yaml` — CLIP `ViT-L-14 / openai` (C4), ResNet-50 `IMAGENET1K_V2` (C1), logistic
  probe (`C: 1.0`, `max_iter: 1000`), and `finetune_resnet` hyperparams (C2).
- `configs/experiment.yaml` — `seeds: [13, 37, 71]`, the JPEG core sweep `[90,70,50,30,10]`, and
  calibration/bootstrap settings.

**Local vs Colab paths:** on local, `configs/data.yaml` uses relative `data/raw/...` and `data/crops`.
On Colab the notebooks override `ROOT = /content/drive/MyDrive/faceproof` and read/write everything
under Drive (so work survives runtime resets). On Kaggle, read datasets from `/kaggle/input/...`.

## 4. Drive persistent-storage layout

The notebooks build and reuse this tree under `MyDrive/faceproof/` (rebuild once, reuse everywhere):

```
MyDrive/faceproof/
├── data/
│   ├── crops/{real,stylegan,sd}/      # 224×224 + JPEG-90 compression-matched faces
│   └── manifest.csv                   # path,label,generator,split (seed-13 splits)
├── models/
│   ├── features/{clip_all,resnet_all}.npy   # cached frozen features (C4 768-d, C1 2048-d)
│   └── probes/{c1_resnet.joblib, c4_clip.joblib, c2_resnet_finetune.pt}
└── reports/
    ├── results.csv                    # one row per (condition,test_gen,corruption,metric)
    ├── table_auroc.md, table_calibration.md
    ├── calib_arrays.npz               # reliability-diagram arrays
    └── figures/*.png
```

## 5. Run order

Run the notebooks in numeric order (each is self-contained: clones the repo at `BRANCH="main"`, mounts
Drive, rebuilds the seed-13 manifest, imports from `src/`). Gates and per-notebook details are in
[`notebooks/README.md`](notebooks/README.md).

1. `01_preprocess_and_leakage` — crops, manifest, DCT leakage check *(gate: matched batch loads;
   leakage ≈ chance)*
2. `02_cache_features` → `03_probe_crossgen` → `04_robustness` → `05_multigenerator` →
   `06_calibration` → `07_seeds_ci` → `08_extensions_c2_c3` → `09_aggregate_results` → `10_figures`
3. Optional: `ablation_resolution` (resolution-matched T2I re-eval — hardens the below-chance finding).

Then: `demo/app.py` (Gradio; see [`demo/README.md`](demo/README.md)) and `reports/report.md`.

## 6. Local (CPU) smoke test

You don't need a GPU to sanity-check the library code:

```bash
pip install -r requirements.txt
python -c "import src.preprocessing, src.features, src.probe, src.metrics, src.corruptions, src.calibration; print('imports OK')"
```

Feature extraction and fine-tuning want a GPU; the probe/metrics/calibration steps run fine on CPU
once features are cached.
