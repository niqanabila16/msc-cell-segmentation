"""
tests/unit/test_data_validation.py — Unit test murni untuk src/data_utils.py
(FR03, FR04, dan sebagian logika decode yang dipakai FR02).

PENTING -- baca ini dulu: hasil E2E (lihat tests/e2e/test_upload_and_validation.py
dan README.md bagian temuan) menunjukkan validate_image_file() dan
validate_label_file() di-import di app.py TAPI TIDAK PERNAH DIPANGGIL.
Jadi unit test di bawah ini membuktikan LOGIKA VALIDASINYA SENDIRI benar
(seandainya nanti dipakai app.py), TAPI TIDAK membuktikan aplikasi yang
sedang berjalan benar-benar menegakkannya -- itu tetap tanggung jawab
test E2E. Dua level ini sengaja dipisah, bukan duplikasi:
  - Unit  : "apakah validate_image_file() mengimplementasikan FR03 dengan benar?"
  - E2E   : "apakah app.py yang berjalan benar-benar menegakkan FR03?"
"""
import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import config
from src.data_utils import (
    load_label_from_bytes,
    validate_image_file,
    validate_label_file,
)


def _image_bytes(width: int, height: int, fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 120, 120))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ── FR03: validate_image_file ────────────────────────────────────────────

def test_valid_jpg_within_limits_passes():
    ok, msg = validate_image_file(_image_bytes(64, 64, "JPEG"), "foto.jpg")
    assert ok, msg


def test_valid_png_within_limits_passes():
    ok, msg = validate_image_file(_image_bytes(64, 64, "PNG"), "foto.png")
    assert ok, msg


def test_unsupported_extension_is_rejected():
    ok, msg = validate_image_file(_image_bytes(64, 64, "BMP"), "foto.bmp")
    assert not ok
    assert "tidak didukung" in msg


def test_oversized_image_is_rejected():
    # Bikin file yang PASTI melebihi config.MAX_FILE_SIZE_BYTES, tanpa
    # perlu tahu nilai persisnya di muka -- baca langsung dari config.
    oversized = b"\xff" * (config.MAX_FILE_SIZE_BYTES + 1024)
    ok, msg = validate_image_file(oversized, "besar.jpg")
    assert not ok
    assert "melebihi batas" in msg


def test_image_smaller_than_minimum_dimension_is_rejected():
    too_small = max(config.MIN_IMAGE_DIM - 1, 1)
    ok, msg = validate_image_file(_image_bytes(too_small, too_small, "PNG"), "kecil.png")
    assert not ok
    assert "terlalu kecil" in msg


def test_corrupt_image_bytes_are_rejected_gracefully():
    """File berekstensi .jpg tapi bukan gambar valid -- tidak boleh raise
    exception mentah, harus ditangani jadi (False, pesan)."""
    ok, msg = validate_image_file(b"bukan data gambar sama sekali", "korup.jpg")
    assert not ok
    assert "Gagal membaca citra" in msg


# ── FR04: validate_label_file ───────────────────────────────────────────

def test_valid_png_label_passes():
    mask = Image.new("L", (64, 64), color=0)
    buf = BytesIO()
    mask.save(buf, format="PNG")
    ok, msg = validate_label_file(buf.getvalue(), "mask.png")
    assert ok, msg


def test_unsupported_label_extension_is_rejected():
    ok, msg = validate_label_file(b"halo", "label.txt")
    assert not ok
    assert "tidak didukung" in msg


def test_oversized_label_is_rejected():
    oversized = b"\x00" * (config.MAX_FILE_SIZE_BYTES + 1024)
    ok, msg = validate_label_file(oversized, "besar.png")
    assert not ok


def test_valid_json_label_passes():
    payload = json.dumps({"images": [], "annotations": []}).encode("utf-8")
    ok, msg = validate_label_file(payload, "annotations.json")
    assert ok, msg


def test_malformed_json_label_is_rejected():
    ok, msg = validate_label_file(b"{not valid json", "annotations.json")
    assert not ok
    assert "Format label tidak valid" in msg


def test_corrupt_png_label_is_rejected():
    ok, msg = validate_label_file(b"bukan png sama sekali" * 5, "mask.png")
    assert not ok


# ── Decode mask (dipakai jalur FR02 setelah pencocokan nama file) ──────

def test_load_png_mask_binarizes_nonzero_pixels():
    arr = np.zeros((10, 10), dtype=np.uint8)
    arr[2:5, 2:5] = 200  # blok terang di tengah
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")

    mask = load_label_from_bytes(buf.getvalue(), "mask.png", image_shape=(10, 10))

    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    assert mask[3, 3] == 1
    assert mask[0, 0] == 0
    assert mask.sum() == 9  # blok 3x3


def test_load_labelme_polygon_json_produces_nonempty_mask():
    payload = {
        "imageHeight": 20,
        "imageWidth": 20,
        "shapes": [
            {"points": [[2, 2], [2, 10], [10, 10], [10, 2]]},
        ],
    }
    mask = load_label_from_bytes(
        json.dumps(payload).encode("utf-8"), "label.json", image_shape=(20, 20)
    )
    assert mask.shape == (20, 20)
    assert mask.sum() > 0
    assert mask[5, 5] == 1  # di dalam kotak 2..10


def test_load_coco_master_json_matches_by_filename_and_decodes_polygon():
    """
    Ini versi formal dari verifikasi yang sebelumnya dijalankan manual
    terhadap fixture Selenium (annotations.json + sel_003.jpg) -- sekarang
    jadi unit test permanen, jalan dalam milidetik, tanpa perlu browser
    ataupun file citra sungguhan.
    """
    coco = {
        "images": [
            {"id": 1, "file_name": "sel_003.jpg", "height": 30, "width": 30},
            {"id": 2, "file_name": "lain.jpg", "height": 30, "width": 30},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "segmentation": [[5, 5, 5, 20, 20, 20, 20, 5]]},
            {"id": 2, "image_id": 2, "segmentation": [[0, 0, 0, 5, 5, 5]]},
        ],
    }
    mask = load_label_from_bytes(
        json.dumps(coco).encode("utf-8"),
        "annotations.json",
        image_shape=(30, 30),
        target_image_name="sel_003.jpg",
    )
    assert mask.shape == (30, 30)
    assert mask.sum() > 0
    assert mask[10, 10] == 1  # di dalam kotak annotation image_id=1
    # Pastikan HANYA anotasi milik image_id=1 yang dipakai (bukan tercampur
    # dengan anotasi image_id=2 di sudut kiri-atas).
    assert mask[1, 1] == 0


def test_load_coco_master_json_unmatched_image_returns_empty_mask():
    """FR02 (negatif): citra yang namanya TIDAK ada di annotations.json
    harus menghasilkan mask kosong, bukan error atau mask tercampur milik
    citra lain."""
    coco = {
        "images": [{"id": 1, "file_name": "ada.jpg", "height": 10, "width": 10}],
        "annotations": [{"id": 1, "image_id": 1, "segmentation": [[1, 1, 1, 5, 5, 5]]}],
    }
    mask = load_label_from_bytes(
        json.dumps(coco).encode("utf-8"),
        "annotations.json",
        image_shape=(10, 10),
        target_image_name="tidak_ada.jpg",
    )
    assert mask.shape == (10, 10)
    assert mask.sum() == 0
