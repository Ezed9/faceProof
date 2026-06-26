"""
Day 1 — Face preprocessing: detect/align/crop + compression matching.

Why this matters: if real and fake images differ in resolution or JPEG history,
a detector can 'cheat' on those instead of synthesis cues. We route EVERY image
through identical alignment, resizing, and JPEG re-encoding so compression cannot
become a label proxy.

Usage (example):
    from src.preprocessing import preprocess_folder
    preprocess_folder("data/raw/ffhq_real", "data/crops/real",
                      image_size=224, jpeg_quality=90)

Falls back to a center-crop if facenet-pytorch (MTCNN) is unavailable, so it runs
even before you install the face detector.
"""
from __future__ import annotations
import io
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# Optional MTCNN face detector/aligner.
try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except Exception:  # pragma: no cover
    _HAS_MTCNN = False

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    """Fallback: square center crop then resize."""
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.BICUBIC)


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Re-encode as JPEG at a fixed quality so all images share compression history."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def preprocess_image(path: str | Path, size: int, jpeg_quality: int, detector=None) -> Image.Image | None:
    """Load -> align/crop -> resize -> compression-match. Returns a PIL image or None."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None

    if detector is not None:
        # MTCNN returns an aligned face tensor; we re-render to a PIL crop.
        face = detector(img)
        if face is None:
            return None
        import numpy as np
        arr = ((face.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype("uint8")
        img = Image.fromarray(arr).resize((size, size), Image.BICUBIC)
    else:
        img = _center_crop_resize(img, size)

    return jpeg_recompress(img, jpeg_quality)


def preprocess_folder(in_dir: str | Path, out_dir: str | Path, image_size: int = 224,
                      jpeg_quality: int = 90, use_mtcnn: bool = True, limit: int | None = None) -> int:
    """Preprocess every image in in_dir into out_dir. Returns the count written."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = None
    if use_mtcnn and _HAS_MTCNN:
        detector = MTCNN(image_size=image_size, margin=20, post_process=False, select_largest=True)
    elif use_mtcnn and not _HAS_MTCNN:
        print("[preprocess] facenet-pytorch not installed; using center-crop fallback.")

    files = [p for p in sorted(in_dir.rglob("*")) if p.suffix.lower() in _IMG_EXT]
    if limit:
        files = files[:limit]

    n = 0
    for p in tqdm(files, desc=f"preprocess {in_dir.name}"):
        out = preprocess_image(p, image_size, jpeg_quality, detector)
        if out is None:
            continue
        out.save(out_dir / f"{p.stem}.jpg", format="JPEG", quality=jpeg_quality)
        n += 1
    print(f"[preprocess] wrote {n}/{len(files)} images to {out_dir}")
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    preprocess_folder(a.in_dir, a.out_dir, a.image_size, a.jpeg_quality, limit=a.limit)
