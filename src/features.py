"""
Days 2-3 — Frozen feature extraction + caching.

Extract image embeddings ONCE from frozen backbones and cache to .npy, so probe
training is instant and you never recompute. Supports:
  - CLIP ViT-L/14 (open_clip)          -> condition C4
  - ImageNet ResNet-50 (torchvision)   -> condition C1

Usage:
    from src.features import extract_clip, extract_resnet
    X, paths = extract_clip(list_of_image_paths, cache="models/features/clip_train.npy")
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch


def _batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.no_grad()
def extract_clip(paths: Sequence[str], backbone="ViT-L-14", pretrained="openai",
                 batch_size=64, cache: str | None = None, device=None) -> np.ndarray:
    """Return [N, 768] CLIP image features (L2-normalized)."""
    if cache and Path(cache).exists():
        print(f"[features] loading cached CLIP features {cache}")
        return np.load(cache)
    import open_clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
    model = model.to(device).eval()

    feats = []
    for batch in tqdm(list(_batched(list(paths), batch_size)), desc="CLIP features"):
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        f = model.encode_image(imgs)
        f = torch.nn.functional.normalize(f, dim=-1)
        feats.append(f.cpu().numpy())
    out = np.concatenate(feats, axis=0)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out


@torch.no_grad()
def extract_resnet(paths: Sequence[str], batch_size=64, cache: str | None = None, device=None) -> np.ndarray:
    """Return [N, 2048] frozen ImageNet ResNet-50 penultimate features."""
    if cache and Path(cache).exists():
        print(f"[features] loading cached ResNet features {cache}")
        return np.load(cache)
    from torchvision import models, transforms
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=weights)
    net.fc = torch.nn.Identity()          # penultimate features
    net = net.to(device).eval()
    tf = weights.transforms()

    feats = []
    for batch in tqdm(list(_batched(list(paths), batch_size)), desc="ResNet features"):
        imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch]).to(device)
        feats.append(net(imgs).cpu().numpy())
    out = np.concatenate(feats, axis=0)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, out)
    return out
