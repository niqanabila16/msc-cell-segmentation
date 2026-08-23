"""
src/inference_engine.py — Mesin inferensi U-Net (v7).

Perubahan dari v6:
- Cabang SMP di run_unet() sekarang RESIZE ke resolusi training persis
  (model._input_h / model._input_w dari checkpoint config, BILINEAR — sama
  seperti SegmentationDataset._load_image() saat training), bukan
  mempertahankan resolusi asli citra upload. Sebelumnya "tidak ada resize"
  untuk SMP membuat skala sel relatif ke receptive field ResNet50 berbeda
  dari training, menghasilkan segmentasi berbercak/noisy walau val_dice saat
  training bagus. Fallback ke scale proporsional (model._input_scale) kalau
  checkpoint tidak menyimpan input_h/input_w eksplisit.

Perubahan dari v5:
- run_unet() sekarang menghormati model._in_channels juga untuk cabang SMP
  ResNet50-UNet, bukan hanya vanilla U-Net:
    * in_channels == 1 -> tensor 1-channel via image_to_tensor_imagenet_gray()
      (grayscale + normalisasi ImageNet-style 1-channel), bukan dipaksa RGB
      seperti sebelumnya. Ini yang menyebabkan RuntimeError "expected input
      ... to have 1 channels, but got 3 channels instead" pada model SMP
      yang dilatih grayscale.
    * in_channels == 3 -> tetap image_to_tensor_imagenet() (RGB) seperti semula.

Perubahan dari v4:
- SAM2 dihapus sepenuhnya (run_sam2, dispatcher SAM2, gray_converted).
- Resize kotak tetap (512x512 / BORDER_REFLECT_101) DIHAPUS untuk vanilla
  U-Net — itu kebutuhan khusus SAM2. Vanilla U-Net di sini fully
  convolutional sehingga bisa penerima ukuran berapa pun; hanya
  diskalakan proporsional sesuai model._input_scale (default dari
  training, biasanya 0.5) agar konsisten dengan distribusi training.

Pipeline saat ini:

    Vanilla UNet (model._smp tidak ada / False):
      preprocessing -> skala proporsional sesuai model._input_scale
      -> tensor 1 atau 3 channel sesuai model._in_channels (/255.0)
      -> argmax(dim=1)
      -> mask_eval di resolusi hasil skala, mask_display upscale ke ukuran asli

    SMP ResNet50-UNet (model._smp == True):
      preprocessing -> resize BILINEAR ke model._input_h/_input_w (resolusi
      training di checkpoint) -> tensor 1 atau 3 channel sesuai
      model._in_channels, normalisasi ImageNet-style (mean/std)
      -> sigmoid > 0.5
      -> mask_eval di resolusi training, mask_display upscale NEAREST ke
      ukuran gambar asli
"""

import time
import numpy as np
import torch
from PIL import Image

from src.image_utils import (
    apply_preprocessing, ensure_rgb,
    image_to_tensor, image_to_tensor_imagenet, image_to_tensor_imagenet_gray,
)


# ── U-Net ─────────────────────────────────────────────────────────────────────

def resize_mask(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize mask segmentasi dengan nearest neighbor.
    target_hw = (height, width)
    """
    h, w = target_hw
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    )


def run_unet(
    model,
    image: np.ndarray,
    preprocessing: str = "none",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Jalankan inferensi U-Net. Mendukung dua arsitektur secara otomatis.

    Deteksi jenis model via atribut model._smp (di-set oleh model_loader.py):

    ┌─────────────────────┬──────────────────────────────────────────────────┐
    │ model._smp = False  │ Vanilla UNet kustom (relu / afpm / sbpiplu)      │
    │ (default)           │ • Skala proporsional sesuai model._input_scale   │
    │                     │ • Channel sesuai model._in_channels (1 atau 3)   │
    │                     │ • Normalisasi: /255.0 -> [0, 1]                  │
    │                     │ • Output: (B, 2, H, W) -> argmax -> mask         │
    │                     │ • mask_display: upscale nearest ke ukuran asli   │
    ├─────────────────────┼──────────────────────────────────────────────────┤
    │ model._smp = True   │ SMP ResNet50-UNet                                │
    │                     │ • Resize ke model._input_h/_input_w (BILINEAR)   │
    │                     │   — persis resolusi training di checkpoint       │
    │                     │ • Channel sesuai model._in_channels (1 atau 3)   │
    │                     │ • Normalisasi: ImageNet mean/std                  │
    │                     │ • Output: (B, 1, H, W) logits -> sigmoid > 0.5   │
    │                     │ • mask_eval: resolusi training; mask_display:    │
    │                     │   upscale NEAREST ke ukuran gambar asli          │
    └─────────────────────┴──────────────────────────────────────────────────┘

    Returns
    -------
    mask_eval    : np.ndarray uint8 — mask di resolusi evaluasi
    mask_display : np.ndarray uint8 — mask di resolusi gambar asli (untuk viz)
    elapsed      : float — waktu inferensi (detik), tidak termasuk preprocessing
    """
    is_smp = getattr(model, '_smp', False)
    in_channels = getattr(model, '_in_channels', 3)
    input_scale = getattr(model, '_input_scale', 1.0)

    image = ensure_rgb(image)
    image = apply_preprocessing(image, preprocessing)

    device = next(model.parameters()).device

    if is_smp:
        # ── SMP ResNet50-UNet ──────────────────────────────────────────────
        # Resize ke resolusi training persis (model._input_h/_input_w dari
        # checkpoint config) dengan BILINEAR — sama seperti
        # SegmentationDataset._load_image() saat training. Sebelumnya cabang
        # ini SAMA SEKALI TIDAK resize, sehingga citra masuk pada resolusi
        # native upload-nya; skala sel relatif ke receptive field ResNet50
        # jadi berbeda dari training dan hasil segmentasi berbercak/noisy
        # meski val_dice saat training bagus. Fallback ke scale proporsional
        # kalau checkpoint tidak menyimpan input_h/input_w eksplisit.
        target_h = getattr(model, '_input_h', None)
        target_w = getattr(model, '_input_w', None)
        h0, w0 = image.shape[:2]

        if target_h and target_w:
            img_r = np.array(
                Image.fromarray(image).resize((target_w, target_h), resample=Image.BILINEAR)
            )
        elif input_scale and input_scale != 1.0:
            new_w = max(1, int(round(w0 * input_scale)))
            new_h = max(1, int(round(h0 * input_scale)))
            img_r = np.array(
                Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR)
            )
        else:
            img_r = image

        # Channel tensor mengikuti model._in_channels — sebelumnya cabang ini
        # selalu memaksa 3-channel (image_to_tensor_imagenet), yang crash
        # untuk checkpoint grayscale (in_channels=1).
        if in_channels == 1:
            tensor = image_to_tensor_imagenet_gray(img_r).to(device)
        else:
            tensor = image_to_tensor_imagenet(img_r).to(device)
    else:
        # ── Vanilla U-Net ──────────────────────────────────────────────────
        # Skala proporsional (bukan pad+resize kotak — itu kebutuhan SAM2),
        # lalu bangun tensor dengan jumlah channel sesuai checkpoint.
        h, w = image.shape[:2]
        if input_scale != 1.0:
            new_w = max(1, int(round(w * input_scale)))
            new_h = max(1, int(round(h * input_scale)))
            img_r = np.array(
                Image.fromarray(image).resize((new_w, new_h), resample=Image.BICUBIC)
            )
        else:
            img_r = image

        if in_channels == 1:
            gray = np.array(Image.fromarray(img_r).convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float().to(device)
        else:
            tensor = image_to_tensor(img_r).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)

        if is_smp:
            # smp.Unet output: (B, classes, H, W) — binary -> (B, 1, H, W) logits
            pred_tensor = (torch.sigmoid(output) > 0.5).squeeze(1).long()  # (B, H, W)
        else:
            # Vanilla U-Net output: (B, 2, H, W) -> argmax
            pred_tensor = torch.argmax(output, dim=1)  # (B, H, W)

    elapsed = time.perf_counter() - t0

    # Mask untuk evaluasi (resolusi model)
    mask_eval = pred_tensor.squeeze(0).cpu().numpy().astype(np.uint8)

    # Mask untuk visualisasi (resolusi gambar asli)
    h_orig, w_orig = image.shape[:2]
    mask_display = resize_mask(mask_eval, (h_orig, w_orig))

    return mask_eval, mask_display, elapsed


# ── Dispatcher ────────────────────────────────────────────────────────────────

def run_inference(
    model_instance,
    image: np.ndarray,
    preprocessing: str = "none",
) -> dict:
    """
    Jalankan inferensi satu citra dengan model U-Net yang sudah dimuat.

    Parameters
    ----------
    model_instance : objek model yang sudah dimuat via model_loader.load_unet()
    image          : np.ndarray uint8 (H, W, 3) atau (H, W)
    preprocessing  : kunci metode preprocessing (default "none")

    Returns
    -------
    dict dengan key:
      mask          : np.ndarray uint8 — mask di resolusi evaluasi
      mask_display  : np.ndarray uint8 — mask di resolusi gambar asli
      time_sec      : float — waktu inferensi
      warnings      : list[str] — pesan peringatan
    """
    mask_eval, mask_display, t = run_unet(model_instance, image, preprocessing)

    return {
        "mask":         mask_eval,
        "mask_display": mask_display,
        "time_sec":     t,
        "warnings":     [],
    }