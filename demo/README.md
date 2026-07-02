# FaceProof demo (Gradio → Hugging Face Spaces, free)

A screening auditor: upload a face → **real / synthetic** with a *temperature-calibrated* confidence
and a reliability note. It is the demo of the research (frozen CLIP ViT-L/14 + a logistic probe =
condition **C4**), **not** a liveness check or a production pipeline.

> **Honest caveat (this is the whole point of the project).** The probe was trained on StyleGAN vs
> real faces. It generalizes to a *near* generator (Stable Diffusion v1.4, AUROC ≈ 0.9) but
> **collapses below chance on modern text-to-image faces (SDXL / Flux / DALL·E 3)** — it will confidently
> call those "real." Treat near-0.5 scores as "needs review," and don't present the demo as a solved
> detector. See [`../reports/report.md`](../reports/report.md).

## Files
- `app.py` — the Gradio app. Embeds the face with frozen CLIP, applies the saved probe, returns
  `P(synthetic)`. Finds `models/probes/c4_clip.joblib` next to itself **or** one level up (so it works
  both as `demo/app.py` in the repo and as `app.py` at a Space root). Falls back to CLIP
  `ViT-L-14 / openai` if `configs/model.yaml` is absent.
- `requirements.txt` — gradio, torch, torchvision, open_clip_torch, scikit-learn, joblib, pyyaml, pillow, numpy, scipy.
- The **probe** `c4_clip.joblib` is **not** in git — copy it from Drive (`MyDrive/faceproof/models/probes/`,
  produced by notebook `03_probe_crossgen`). It is tiny (a logistic head), unlike the datasets/features.

## Run locally
```bash
pip install -r demo/requirements.txt
# ensure the probe exists at models/probes/c4_clip.joblib (copy from Drive)
python demo/app.py
```

## Optional: calibrated confidence (temperature scaling)
The app multiplies the logit by `1/T` before the sigmoid. Without a temperature file it uses `T = 1.0`
(uncalibrated). To ship the in-distribution calibration from Day 6, create:

`models/probes/c4_clip_temperature.json`
```json
{"T": <value>}
```
where `<value>` is the temperature `fit_temperature(...)` returns for **C4 on the in-distribution
validation set** in notebook `06_calibration` (the one that took in-dist ECE 0.020 → 0.009). Note this
only calibrates *in-distribution*: our H3 finding is that in-dist temperature scaling does **not** fix
CLIP's over-confidence under generator shift — the reliability note in the UI says so.

## Deploy to a free Hugging Face Space
1. Create a new **Space** → SDK **Gradio** → free CPU hardware is enough (the probe is tiny; CLIP runs
   on CPU, a few seconds per image).
2. Put these at the **Space root**: `app.py`, `requirements.txt`, and `models/probes/c4_clip.joblib`
   (+ optional `models/probes/c4_clip_temperature.json`). No datasets, no cached features, no configs needed.
3. Add a Space `README.md` with this YAML front matter so Spaces knows how to launch it:
   ```yaml
   ---
   title: FaceProof
   emoji: 🕵️
   colorFrom: indigo
   colorTo: gray
   sdk: gradio
   app_file: app.py
   pinned: false
   ---
   ```
4. Push (via the Space's git remote or the web uploader):
   ```bash
   git add app.py requirements.txt models/probes/c4_clip.joblib README.md
   git commit -m "FaceProof demo"
   git push        # to the Space remote
   ```
5. The Space builds from `requirements.txt` and launches `app.py`. First build downloads the CLIP
   weights (~1.7 GB) — subsequent loads are cached.

**Note:** copying the probe from Drive and pushing to your Space are manual steps (they need your Drive
file and your Hugging Face account); the app code and deploy config here are ready.
