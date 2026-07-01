"""
FaceProof demo — Gradio app for Hugging Face Spaces (Day 11).

Loads the saved CLIP probe (the small logistic head only — NOT the datasets) and classifies an
uploaded face as real / synthetic with a *calibrated* confidence. This is a screening auditor, not
a liveness check; it is the demo of the research, not a production pipeline.

Run locally:  python demo/app.py
The probe lives at models/probes/c4_clip.joblib (ship this file with the Space; it is tiny).
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import open_clip
import torch
import yaml
from PIL import Image

# Resolve artifacts whether app.py runs from the repo (demo/app.py -> repo root holds models/) or
# from a Hugging Face Space root (app.py + models/ side by side). First existing candidate wins.
_HERE = Path(__file__).resolve().parent
_ROOTS = [_HERE, _HERE.parent]


def _find(rel: str) -> Path:
    for base in _ROOTS:
        p = base / rel
        if p.exists():
            return p
    return _ROOTS[0] / rel  # default location (may not exist yet)


PROBE_PATH = _find("models/probes/c4_clip.joblib")
TEMP_PATH = _find("models/probes/c4_clip_temperature.json")  # optional {"T": <float>}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_clip() -> tuple[torch.nn.Module, callable]:
    cfg_path = _find("configs/model.yaml")
    if cfg_path.exists():
        clip_cfg = yaml.safe_load(open(cfg_path))["clip"]
        backbone, pretrained = clip_cfg["backbone"], clip_cfg["pretrained"]
    else:
        backbone, pretrained = "ViT-L-14", "openai"  # C4 defaults (match src/features.py)
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    model = model.to(DEVICE).eval()
    return model, preprocess


def _load_temperature() -> float:
    if TEMP_PATH.exists():
        return float(json.loads(TEMP_PATH.read_text())["T"])
    return 1.0


if not PROBE_PATH.exists():
    raise FileNotFoundError(
        f"Probe not found at {PROBE_PATH}. Train it in notebook 03 and copy "
        "models/probes/c4_clip.joblib into the Space before launching."
    )

_MODEL, _PREPROCESS = _load_clip()
_PROBE = joblib.load(PROBE_PATH)
_T = _load_temperature()


@torch.no_grad()
def _embed(img: Image.Image) -> np.ndarray:
    x = _PREPROCESS(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    feat = _MODEL.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()


def classify(img: Image.Image) -> tuple[dict, str]:
    if img is None:
        return {}, "Upload a face image."
    feat = _embed(img)
    logit = float(_PROBE.decision_function(feat)[0])
    p_synth = float(1.0 / (1.0 + np.exp(-logit / _T)))  # temperature-calibrated
    label = "SYNTHETIC (flagged)" if p_synth >= 0.5 else "real (passed)"
    note = ("Confidence is temperature-calibrated. Reliability degrades on generators unseen in "
            "training and under heavy compression — treat near-0.5 scores as 'needs review'.")
    return {"synthetic": p_synth, "real": 1 - p_synth}, f"**{label}**\n\n{note}"


with gr.Blocks(title="FaceProof — synthetic-face auditor") as demo:
    gr.Markdown("# FaceProof\nScreening auditor for synthetic-identity injection at enrollment. "
                "Research demo (frozen CLIP + logistic probe) — not a production or liveness system.")
    with gr.Row():
        inp = gr.Image(type="pil", label="Face image")
        with gr.Column():
            out_lab = gr.Label(label="Probability")
            out_txt = gr.Markdown()
    inp.change(classify, inputs=inp, outputs=[out_lab, out_txt])


if __name__ == "__main__":
    demo.launch()
