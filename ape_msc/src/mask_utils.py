"""src/mask_utils.py — Operasi pada mask biner (v3)."""
from typing import Tuple
import numpy as np
import cv2


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mask > threshold).astype(np.uint8)


def compute_tp_fp_fn(pred: np.ndarray, gt: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, g = pred.astype(bool), gt.astype(bool)
    return p & g, p & ~g, ~p & g


def combine_masks_union(masks: list) -> np.ndarray:
    if not masks:
        raise ValueError("List mask kosong.")
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out = np.logical_or(out, m > 0).astype(np.uint8)
    return out


def postprocess_mask(mask: np.ndarray, min_area: int = 50) -> np.ndarray:
    """Hapus komponen koneksi kecil (noise)."""
    u8 = (mask > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(u8)
    clean = np.zeros_like(u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return (clean > 0).astype(np.uint8)
