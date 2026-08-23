"""
src/image_utils.py — preprocessing dan resize citra.

Fokus utama bagian polynomial:
- Stage 1: fit background + divide
- Stage 2: stage 1 + normalize output ke [0, 1]

Koordinat piksel dipetakan ke [-1, 1] sebelum fitting untuk menjaga
stabilitas numerik pada least-squares polynomial background fitting.

Perubahan v6:
- Dihapus: _pad_to_square, resize_for_unet, resize_for_sam2, upscale_mask.
  Semua ini adalah kebutuhan pad+resize-kotak / resize langsung khusus SAM2
  (lihat catatan v7 di app.py). Resize mask aktual untuk U-Net sekarang
  ditangani oleh resize_mask() di inference_engine.py.
- Ditambah: image_to_tensor_imagenet_gray() untuk SMP ResNet50-UNet yang
  dilatih dengan checkpoint in_channels=1 (grayscale). Sebelumnya
  image_to_tensor_imagenet() hanya mendukung 3-channel, menyebabkan
  RuntimeError channel mismatch saat model SMP grayscale menerima tensor
  RGB paksa.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np
from numpy.polynomial.polynomial import polyvander2d
from PIL import Image

import config


# -----------------------------------------------------------------------------
# Channel helpers
# -----------------------------------------------------------------------------

def is_grayscale(image: np.ndarray) -> bool:
    """Return True jika image 2D atau 3D dengan channel tunggal."""
    if len(image.shape) == 2:
        return True
    if len(image.shape) == 3 and image.shape[2] == 1:
        return True
    return False


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Pastikan image berada pada format RGB 3-channel."""
    if is_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


# -----------------------------------------------------------------------------
# Labels untuk UI — 2 stage
# -----------------------------------------------------------------------------

POLY_STAGE_LABELS: Dict[int, str] = {
    1: "Stage 1 — Fit + Divide (Dasar)",
    2: "Stage 2 — + Normalize Output ke [0, 1] (Full)",
}

POLY_STAGE_DESC: Dict[int, str] = {
    1: (
        "Polynomial surface fitting via least-squares (orde 3, 10 basis term), "
        "diikuti pembagian piksel dengan background yang diestimasi. "
        "Mengikuti pendekatan dasar Li et al. (2025) dan Lee et al. (2014). "
        "Output tidak terikat di [0, 1] — nilai > 1.0 dapat muncul di area sel-dense."
    ),
    2: (
        "Tambahan normalize_to_01 setelah pembagian. Memastikan output terikat "
        "dalam [0, 1] agar percentile normalization berikutnya menghitung p99 "
        "dari distribusi yang bermakna secara biologis."
    ),
}

POLY_STAGE_ROLES: Dict[int, str] = {
    1: (
        "Mengestimasi dan mengoreksi distribusi iluminasi background. "
        "Tanpa koreksi ini, gradien kecerahan spasial dari vignetting dapat "
        "ikut terbawa ke pipeline berikutnya."
    ),
    2: (
        "Memastikan kompatibilitas output dengan tahap percentile normalization. "
        "Tanpa batas ini, nilai > 1.0 yang tersisa dapat mengganggu clipping 1–99."
    ),
}


# -----------------------------------------------------------------------------
# Preprocessing core
# -----------------------------------------------------------------------------

def _normalize_to_01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi - lo < eps:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - lo) / (hi - lo)).astype(np.float32)


def rgb_to_gray(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Konversi RGB uint8 (H, W, 3) ke grayscale float32 [0.0, 1.0].
    Bobot luminance ITU-R BT.709: Y = 0.2126R + 0.7152G + 0.0722B.
    """
    rgb_01 = rgb_uint8.astype(np.float32) / 255.0
    r, g, b = rgb_01[..., 0], rgb_01[..., 1], rgb_01[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _poly_design_matrix_2d(xx: np.ndarray, yy: np.ndarray, order: int) -> np.ndarray:
    """
    Bangun matriks desain polynomial 2D dari koordinat xx, yy.
    Hanya term dengan total degree i + j <= order yang dipakai.
    """
    vand = polyvander2d(xx.ravel(), yy.ravel(), [order, order])
    vand = vand.reshape(-1, (order + 1) ** 2)

    mask = np.array(
        [(i + j) <= order for i in range(order + 1) for j in range(order + 1)],
        dtype=bool,
    )
    return vand[:, mask]


def apply_poly_background(
    gray_01: np.ndarray,
    stage: Optional[int] = None,
) -> np.ndarray:
    """
    Polynomial background correction orde-3.

    Alur:
    1) Koordinat dipetakan ke [-1, 1] untuk stabilitas numerik.
    2) Permukaan background di-fit dengan least squares polynomial 2D.
    3) Citra asli dibagi dengan background hasil estimasi.
    4) Jika stage >= 2, output dinormalisasi lagi ke [0, 1].
    """
    if stage is None:
        stage = int(getattr(config, "POLY_STAGE", 2))

    order = 3
    h, w = gray_01.shape

    # 1) Koordinat piksel dipetakan ke [-1, 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx = 2.0 * (xx / max(w - 1, 1)) - 1.0
    yy = 2.0 * (yy / max(h - 1, 1)) - 1.0

    # 2) Matriks desain polynomial 2D orde-3
    A = _poly_design_matrix_2d(xx, yy, order)
    b = gray_01.astype(np.float64).ravel()

    # 3) Least squares untuk koefisien background
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    background = (A @ coeffs).reshape(h, w).astype(np.float32)

    # Hindari pembagian dengan nol
    background = np.clip(background, 1e-8, None)

    # 4) Koreksi citra
    corrected = gray_01.astype(np.float32) / background
    corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0)
    corrected = np.clip(corrected, 0.0, None)

    # 5) Normalisasi output akhir
    if stage >= 2:
        corrected = _normalize_to_01(corrected)

    return corrected.astype(np.float32)


def get_poly_intermediate_stages(
    gray_01: np.ndarray,
    stage_max: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Ambil citra intermediate untuk kebutuhan visualisasi/laporan."""
    if stage_max is None:
        stage_max = int(getattr(config, "POLY_STAGE", 2))

    stages: Dict[str, np.ndarray] = {}
    stages["Stage 0 — Input"] = gray_01.astype(np.float32)

    for s in range(1, stage_max + 1):
        result = apply_poly_background(gray_01, stage=s)
        vis = _normalize_to_01(result) if result.max() > 1.0 else result
        stages[POLY_STAGE_LABELS[s]] = vis.astype(np.float32)

    return stages


def apply_percentile_clip(
    gray: np.ndarray,
    low: float = None,
    high: float = None,
) -> np.ndarray:
    """
    Percentile normalization: clip [p_low, p_high] lalu rescale ke [0, 1].
    """
    if low is None:
        low = getattr(config, "PERCENTILE_LOW", 1.0)
    if high is None:
        high = getattr(config, "PERCENTILE_HIGH", 99.0)

    x = gray.astype(np.float32)
    p_low = float(np.percentile(x, low))
    p_high = float(np.percentile(x, high))

    if p_high <= p_low:
        return _normalize_to_01(x)

    clipped = np.clip(x, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def _apply_pipeline_steps(gray_01: np.ndarray, steps: List[str]) -> np.ndarray:
    x = gray_01.astype(np.float32)
    for step in steps:
        if step == "poly":
            x = apply_poly_background(x)
        elif step == "percentile":
            x = apply_percentile_clip(x)
        else:
            raise ValueError(f"Langkah pipeline tidak dikenal: '{step}'")
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def apply_gray_pipeline(image: np.ndarray, steps: List[str]) -> np.ndarray:
    gray = rgb_to_gray(ensure_rgb(image))
    x = _apply_pipeline_steps(gray, steps)
    out_u8 = np.clip(x * 255.0, 0, 255).round().astype(np.uint8)
    return np.stack([out_u8, out_u8, out_u8], axis=-1)


# -----------------------------------------------------------------------------
# Preprocessing options — 4 opsi
# -----------------------------------------------------------------------------

PREPROCESSING_OPTIONS: Dict[str, str] = {
    "Tanpa Preprocessing": "none",
    "Polynomial Background Correction": "poly",
    "Percentile Normalization": "percentile",
    "Polynomial + Percentile": "poly_percentile",
}

_GRAY_PIPELINES: Dict[str, List[str]] = {
    "none": [],
    "poly": ["poly"],
    "percentile": ["percentile"],
    "poly_percentile": ["poly", "percentile"],
}


def apply_preprocessing(image: np.ndarray, method: str) -> np.ndarray:
    img = ensure_rgb(image)
    if method in ("none", "", None):
        return img
    if method in ("normalization", "minmax"):
        method = "percentile"
    if method in ("poly3", "poly3basic", "poly3full", "poly2"):
        method = "poly"
    if method in _GRAY_PIPELINES:
        return apply_gray_pipeline(img, _GRAY_PIPELINES[method])
    raise ValueError(
        f"Metode tidak dikenal: '{method}'. Pilihan: {list(_GRAY_PIPELINES.keys())}"
    )


# -----------------------------------------------------------------------------
# Tensor utils
# -----------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Varian 1-channel untuk SMP ResNet50-UNet dengan checkpoint in_channels=1.
# NOTE/TODO: nilai ini adalah rata-rata channel ImageNet, dipakai sebagai
# asumsi default karena encoder SMP dibangun dengan encoder_weights=None
# (dilatih from scratch, bukan pretrained). Ganti dengan mean/std yang
# benar-benar dipakai SegmentationDataset saat training grayscale kalau
# berbeda (mis. murni /255.0 tanpa normalisasi, atau statistik dataset MSC
# sendiri) — kalau tidak, model tidak akan crash tapi prediksi bisa
# diam-diam menyimpang dari performa saat training.
IMAGENET_MEAN_GRAY = float(IMAGENET_MEAN.mean())  # ≈ 0.449
IMAGENET_STD_GRAY = float(IMAGENET_STD.mean())    # ≈ 0.226


def image_to_tensor(image: np.ndarray):
    import torch

    img = image.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def image_to_tensor_imagenet(image: np.ndarray):
    import torch

    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def image_to_tensor_imagenet_gray(image: np.ndarray):
    """
    Bangun tensor 1-channel untuk SMP ResNet50-UNet dengan checkpoint
    in_channels=1. `image` diharapkan RGB uint8 (H, W, 3) — dikonversi ke
    grayscale luminance dulu, baru dinormalisasi.
    """
    import torch

    gray = rgb_to_gray(image)  # (H, W) float32 [0, 1]
    gray = (gray - IMAGENET_MEAN_GRAY) / IMAGENET_STD_GRAY
    return torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float()


def tensor_to_prob(tensor) -> np.ndarray:
    import torch

    with torch.no_grad():
        return torch.sigmoid(tensor.squeeze()).cpu().numpy().astype(np.float32)