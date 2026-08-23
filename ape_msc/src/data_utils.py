"""src/data_utils.py — Validasi dan loading data (COCO RLE support)."""
import json
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config


# ── Validasi (tidak berubah) ──────────────────────────────────────────────────

def validate_image_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_IMAGE_EXTS:
        return False, f"Ekstensi '{ext}' tidak didukung. Gunakan: .jpg .jpeg .png"
    if len(file_bytes) > config.MAX_FILE_SIZE_BYTES:
        mb = len(file_bytes) / (1024 * 1024)
        return False, f"Ukuran {mb:.2f} MB melebihi batas {config.MAX_FILE_SIZE_MB} MB."
    try:
        img = Image.open(BytesIO(file_bytes))
        w, h = img.size
        if w < config.MIN_IMAGE_DIM or h < config.MIN_IMAGE_DIM:
            return False, f"Dimensi {w}×{h} terlalu kecil (min {config.MIN_IMAGE_DIM}px)."
    except Exception as e:
        return False, f"Gagal membaca citra: {e}"
    return True, "OK"


def validate_label_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_LABEL_EXTS:
        return False, f"Ekstensi label '{ext}' tidak didukung."
    if len(file_bytes) > config.MAX_FILE_SIZE_BYTES:
        return False, f"Ukuran label melebihi {config.MAX_FILE_SIZE_MB} MB."
    try:
        if ext == ".png":
            Image.open(BytesIO(file_bytes)).verify()
        else:
            json.loads(file_bytes.decode("utf-8"))
    except Exception as e:
        return False, f"Format label tidak valid: {e}"
    return True, "OK"

def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(BytesIO(file_bytes)).convert("RGB"))

# ── FUNGSI 1 YANG DIUBAH: load_label_from_bytes ───────────────────────────────
# Perubahan:
#   - Tambah parameter `target_image_name: Optional[str] = None`
#   - Teruskan ke _load_json_mask jika file adalah .json

def load_label_from_bytes(
    file_bytes: bytes,
    filename: str,
    image_shape: Tuple[int, int],
    target_image_name: Optional[str] = None,   # ← BARU
) -> np.ndarray:
    """
    Load ground truth mask.
    - .png  → langsung decode sebagai binary mask
    - .json → COCO RLE / Polygon / Labelme
               Jika target_image_name diisi, hanya ambil anotasi
               milik gambar tersebut (mode COCO master JSON).
               Jika None, proses semua anotasi (perilaku lama).
    """
    ext = Path(filename).suffix.lower()
    if ext == ".png":
        return _load_png_mask(file_bytes)
    return _load_json_mask(file_bytes, image_shape, target_image_name)


# ── Tidak berubah ─────────────────────────────────────────────────────────────

def _load_png_mask(file_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(file_bytes)).convert("L")
    return (np.array(img) > 0).astype(np.uint8)


# ── FUNGSI 2 YANG DIUBAH: _load_json_mask ────────────────────────────────────
# Perubahan:
#   - Tambah parameter `target_image_name: Optional[str] = None`
#   - Jika target_image_name diisi → mode COCO master:
#       1. Cari image_id lewat images[] dengan cocokkan file_name stem
#       2. Filter annotations[] hanya yang punya image_id itu
#       3. Decode setiap RLE (handle counts berupa str ATAU bytes)
#          lalu np.maximum untuk union semua instance mask
#   - Jika target_image_name kosong → perilaku lama (proses semua anotasi)
#   - Fix: counts bisa berupa str (dari JSON) → encode ke bytes dulu

def _load_json_mask(
    file_bytes: bytes,
    image_shape: Tuple[int, int],
    target_image_name: Optional[str] = None,   
) -> np.ndarray:
    """
    Dukung:
    - Labelme polygon
    - COCO Polygon
    - COCO RLE (termasuk master annotations.json dengan banyak gambar)
    """
    from pycocotools import mask as maskUtils

    data = json.loads(file_bytes.decode("utf-8"))
    h, w = image_shape

    # ── Deteksi dimensi dari metadata JSON (Labelme / COCO single) ──────────
    if "imageHeight" in data:
        h, w = data["imageHeight"], data["imageWidth"]

    mask = np.zeros((h, w), dtype=np.uint8)
    polys = []

    # ── Labelme ──────────────────────────────────────────────────────────────
    if "shapes" in data:
        for s in data["shapes"]:
            pts = s.get("points", [])
            if pts:
                polys.append(pts)

    # ── COCO (polygon sederhana non-master) ───────────────────────────────────
    elif "polygons" in data:
        polys = data["polygons"]

    # ── COCO master (annotations.json dengan banyak gambar) ──────────────────
    elif "annotations" in data:

        annotations_to_process = data["annotations"]

        # MODE BARU: filter per gambar jika target_image_name diisi
        if target_image_name is not None:
            target_stem = Path(target_image_name).stem

            # Langkah 1: cari image_id yang sesuai di images[]
            image_id = None
            for img_entry in data.get("images", []):
                if Path(img_entry["file_name"]).stem == target_stem:
                    image_id = img_entry["id"]
                    # Gunakan dimensi dari JSON (lebih akurat dari image_shape argumen)
                    h = img_entry["height"]
                    w = img_entry["width"]
                    break

            if image_id is None:
                # Gambar tidak ditemukan dalam JSON → kembalikan mask kosong
                # (bukan raise, supaya tidak crash batch inferensi)
                return np.zeros(image_shape, dtype=np.uint8)

            # Langkah 2: filter anotasi hanya milik image_id ini
            annotations_to_process = [
                a for a in data["annotations"]
                if a["image_id"] == image_id
            ]

            # Re-inisialisasi mask dengan dimensi dari JSON
            mask = np.zeros((h, w), dtype=np.uint8)

        # Langkah 3: proses setiap anotasi (RLE atau Polygon)
        for ann in annotations_to_process:
            seg = ann.get("segmentation", None)
            if not seg:
                continue

            # ── RLE (output dari skripmu) ─────────────────────────────────
            if isinstance(seg, dict) and "counts" in seg:
                rle = seg.copy()

                # FIX PENTING: JSON menyimpan counts sebagai str,
                # tapi pycocotools.decode() butuh bytes
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("ascii")

                decoded = maskUtils.decode(rle)  # shape: (H, W), nilai 0/1
                mask = np.maximum(mask, decoded)

            # ── Polygon (format COCO standar lain) ───────────────────────
            elif isinstance(seg, list):
                for poly in seg:
                    pts = [(poly[i], poly[i + 1])
                           for i in range(0, len(poly) - 1, 2)]
                    polys.append(pts)

    # ── Gambar polygon (Labelme / COCO Polygon) ───────────────────────────────
    for pts in polys:
        flat = np.array(
            [(float(p[0]), float(p[1])) for p in pts], dtype=np.int32
        )
        if len(flat) >= 3:
            cv2.fillPoly(mask, [flat], 1)

    # ── Resize jika dimensi tidak cocok dengan image_shape ───────────────────
    # (Jarang terjadi jika padding sudah konsisten, tapi aman untuk ditambah)
    if mask.shape != image_shape:
        mask = cv2.resize(
            mask, (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    return mask.astype(np.uint8)