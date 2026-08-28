"""
utils/test_data_generator.py — Bangkitkan data uji sintetis.

Dibangkitkan lewat kode (bukan file statis di repo) supaya:
- Tidak perlu commit file biner besar.
- Ukuran file oversize bisa presisi sesuai batas yang diuji.
- Skenario pencocokan nama file bisa dibuat sesuai TIGA pola yang benar-benar
  didukung _parse_uploads() di app.py:
    1) <stem>.png                  (match langsung)
    2) <stem>_mask.png             (match dengan sufiks "_mask")
    3) annotations.json (COCO)     (match lewat images[].file_name + image_id)

Untuk skenario (3), segmentation memakai format POLYGON (list flat
[x1,y1,x2,y2,...]), BUKAN RLE — supaya generator ini tidak perlu
dependency pycocotools (yang butuh compiler C) hanya untuk membuat data
uji. Aplikasi & pycocotools tetap dipakai apa adanya saat men-decode;
polygon adalah cabang kode yang sama sahnya di _load_json_mask()
(`elif isinstance(seg, list)`).
"""
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from testconfig import settings


def _synthetic_cell_image(size=(512, 512), n_blobs=25, seed=None) -> Image.Image:
    """Citra mikroskopis sintetis: latar abu-abu + bercak terang mirip sel."""
    rng = random.Random(seed)
    w, h = size
    base = np.random.default_rng(seed).integers(35, 55, size=(h, w), dtype=np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(n_blobs):
        r = rng.randint(8, 22)
        cx = rng.randint(r, w - r)
        cy = rng.randint(r, h - r)
        shade = rng.randint(180, 240)
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(shade, shade, shade))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def _mask_from_image(img: Image.Image, threshold: int = 150) -> Image.Image:
    """Turunkan ground-truth mask biner dari citra sintetis (blob = foreground)."""
    gray = np.array(img.convert("L"))
    mask = (gray > threshold).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def _pad_file_to_size(path: str, target_bytes: int) -> None:
    """
    Tambahkan byte acak di akhir file supaya ukurannya melewati target_bytes.
    """
    current = os.path.getsize(path)
    if current >= target_bytes:
        return
    with open(path, "ab") as f:
        f.write(os.urandom(target_bytes - current + 2048))


def ensure_test_data(out_dir: str = None) -> dict:
    """Bangkitkan seluruh data uji. Aman dipanggil berkali-kali (idempotent)."""
    out_dir = out_dir or settings.TEST_DATA_DIR
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    invalid_dir = os.path.join(out_dir, "invalid")
    for d in (images_dir, labels_dir, invalid_dir):
        os.makedirs(d, exist_ok=True)

    paths: dict = {}

    # ── Pola 1 — FR02: stem sama persis (sel_001.jpg <-> sel_001.png) ────
    stem = "sel_001"
    img = _synthetic_cell_image(seed=1)
    mask = _mask_from_image(img)
    img_path = os.path.join(images_dir, f"{stem}.jpg")
    label_path = os.path.join(labels_dir, f"{stem}.png")
    img.save(img_path, "JPEG", quality=92)
    mask.save(label_path, "PNG")
    paths.update(matched_image=img_path, matched_label=label_path, matched_stem=stem)

    # ── Pola 2 — FR02: sufiks "_mask" (sel_002.jpg <-> sel_002_mask.png) ─
    stem_mask = "sel_002"
    img2 = _synthetic_cell_image(seed=2)
    mask2 = _mask_from_image(img2)
    img2_path = os.path.join(images_dir, f"{stem_mask}.png")
    label2_path = os.path.join(labels_dir, f"{stem_mask}_mask.png")
    img2.save(img2_path, "PNG")
    mask2.save(label2_path, "PNG")
    paths.update(
        mask_suffix_image=img2_path, mask_suffix_label=label2_path, mask_suffix_stem=stem_mask
    )

    # ── Pola 3 — FR02: master COCO annotations.json (polygon) ───────────
    stem_json = "sel_003"
    img3 = _synthetic_cell_image(seed=3)
    w3, h3 = img3.size
    img3_path = os.path.join(images_dir, f"{stem_json}.jpg")
    img3.save(img3_path, "JPEG", quality=92)
    # Poligon segitiga besar di tengah citra (>=3 titik, format flat [x,y,x,y,...])
    polygon = [
        w3 * 0.2, h3 * 0.8,
        w3 * 0.5, h3 * 0.2,
        w3 * 0.8, h3 * 0.8,
    ]
    coco_json = {
        "images": [{"id": 1, "file_name": f"{stem_json}.jpg", "height": h3, "width": w3}],
        "annotations": [{"id": 1, "image_id": 1, "segmentation": [polygon]}],
    }
    json_path = os.path.join(labels_dir, "annotations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f)
    paths.update(json_image=img3_path, json_master_label=json_path, json_stem=stem_json)

    # ── FR02 (negatif): citra TANPA pasangan label sama sekali ──────────
    stem_unmatched = "sel_999_no_label"
    img_unmatched = _synthetic_cell_image(seed=99)
    img_unmatched_path = os.path.join(images_dir, f"{stem_unmatched}.jpg")
    img_unmatched.save(img_unmatched_path, "JPEG", quality=90)
    paths.update(unmatched_image=img_unmatched_path, unmatched_stem=stem_unmatched)

    # ── FR03: format citra tidak didukung (.bmp, di luar jpg/jpeg/png) ──
    bad_fmt_image = os.path.join(invalid_dir, "invalid_format_image.bmp")
    _synthetic_cell_image(seed=4).save(bad_fmt_image, "BMP")
    paths["invalid_format_image"] = bad_fmt_image

    # ── FR03: citra valid tapi > 10 MB ───────────────────────────────────
    oversized_image = os.path.join(invalid_dir, "oversized_image.jpg")
    _synthetic_cell_image(size=(800, 800), seed=5).save(oversized_image, "JPEG", quality=95)
    _pad_file_to_size(oversized_image, settings.MAX_FILE_SIZE_MB * 1024 * 1024)
    paths["oversized_image"] = oversized_image

    # ── FR04: format label tidak didukung (bukan .png/.json, mis. .txt) ─
    bad_fmt_label = os.path.join(invalid_dir, "invalid_format_label.txt")
    with open(bad_fmt_label, "w") as f:
        f.write("bukan file mask yang valid")
    paths["invalid_format_label"] = bad_fmt_label

    # ── FR04: label .png valid tapi > 10 MB ──────────────────────────────
    oversized_label = os.path.join(invalid_dir, "oversized_label.png")
    _mask_from_image(_synthetic_cell_image(size=(800, 800), seed=6)).save(oversized_label, "PNG")
    _pad_file_to_size(oversized_label, settings.MAX_FILE_SIZE_MB * 1024 * 1024)
    paths["oversized_label"] = oversized_label

    # ── FR04: file berekstensi .png tapi isinya bukan gambar valid ──────
    # Dipasangkan dengan stem yang SAMA dengan citra tervalidasi (sel_001)
    # supaya _parse_uploads() benar-benar mencoba men-decode-nya dan
    # memicu pesan "Gagal memuat mask PNG ...".
    corrupt_label = os.path.join(invalid_dir, f"{stem}_corrupt.png")
    with open(corrupt_label, "wb") as f:
        f.write(b"not a real png file" * 50)
    paths["corrupt_label"] = corrupt_label
    # Salinan dengan nama TEPAT match ke citra 'matched_image' (sel_001.png)
    # supaya bisa dipakai sendirian menggantikan label yang valid.
    corrupt_label_matched_name = os.path.join(invalid_dir, f"{stem}.png")
    with open(corrupt_label_matched_name, "wb") as f:
        f.write(b"not a real png file" * 50)
    paths["corrupt_label_matched_name"] = corrupt_label_matched_name

    paths.update(images_dir=images_dir, labels_dir=labels_dir, invalid_dir=invalid_dir)
    return paths


if __name__ == "__main__":
    generated = ensure_test_data()
    for k, v in generated.items():
        print(f"{k}: {v}")
