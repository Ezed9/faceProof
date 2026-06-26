"""
Day 4 — Real-world degradation transforms for the robustness sweep (RQ2).
Apply these to TEST images only, at increasing strength, and measure AUROC vs strength.
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image, ImageFilter


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_factor(img: Image.Image, factor: float) -> Image.Image:
    """Downscale by factor then back up — simulates resolution loss."""
    if factor >= 1.0:
        return img
    w, h = img.size
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def blur(img: Image.Image, sigma: float) -> Image.Image:
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(img: Image.Image, std: float) -> Image.Image:
    if std <= 0:
        return img
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr = arr + np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


CORRUPTIONS = {"jpeg": jpeg, "resize": resize_factor, "blur": blur, "noise": gaussian_noise}
