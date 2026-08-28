"""
tests/unit/test_metrics.py — Unit test murni untuk src/metrics.py (FR06).

Kenapa ini unit test, BUKAN Selenium: rumus IoU/Dice/Precision/Recall/
konfluensi adalah fungsi Python biasa (numpy in, float out), tidak
menyentuh Streamlit atau browser sama sekali. Menjalankannya lewat
Selenium (upload citra asli -> tunggu inferensi U-Net -> baca angka dari
DOM) hanya bisa memverifikasi angkanya "masuk akal" (0..1), TIDAK bisa
memverifikasi angkanya BENAR -- karena hasil prediksi model terhadap
citra sungguhan tidak diketahui presisi lebih dulu.

Di sini kita kontrol TP/FP/FN secara eksak lewat array buatan tangan,
sehingga nilai IoU/Dice/Precision/Recall yang diharapkan bisa dihitung
manual dan dicocokkan persis -- inilah yang disebut "generating
application state" / pengujian di level yang paling dekat dengan logika
(lihat README.md untuk sumber & alasan lengkap).

Referensi konsep: Martin Fowler, "TestPyramid" (martinfowler.com/bliki/
TestPyramid.html) -- unit test di lapisan paling bawah piramida karena
paling cepat dan paling presisi untuk memvalidasi logika murni.
"""
import numpy as np
import pytest

from src.metrics import (
    average_metrics,
    compute_all_metrics,
    confluence_percent,
    dice_score,
    iou_score,
    percentage_error,
    precision_score,
    recall_score,
)


# ── Fixture: array 4x4 dengan TP/FP/FN yang diketahui persis ────────────
#
#   gt   (baris0-1, kolom0-1): 2x2 blok  -> 4 piksel foreground
#   pred (baris0-1, kolom1-2): 2x2 blok, digeser 1 kolom -> 4 piksel foreground
#
#   Irisan (TP) = kolom1 baris0-1           = 2 piksel
#   FP (pred - gt) = kolom2 baris0-1        = 2 piksel
#   FN (gt - pred) = kolom0 baris0-1        = 2 piksel
#
#   precision = TP/(TP+FP) = 2/4 = 0.5
#   recall    = TP/(TP+FN) = 2/4 = 0.5
#   dice      = 2*P*R/(P+R) = 2*0.5*0.5/1.0 = 0.5
#   iou       = TP/(TP+FP+FN) = 2/6 = 0.333...

@pytest.fixture
def known_masks():
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[0:2, 0:2] = 1
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0:2, 1:3] = 1
    return pred, gt


def test_precision_matches_hand_calculation(known_masks):
    pred, gt = known_masks
    assert precision_score(pred, gt) == pytest.approx(0.5)


def test_recall_matches_hand_calculation(known_masks):
    pred, gt = known_masks
    assert recall_score(pred, gt) == pytest.approx(0.5)


def test_dice_matches_hand_calculation(known_masks):
    pred, gt = known_masks
    assert dice_score(pred, gt) == pytest.approx(0.5)


def test_iou_matches_hand_calculation(known_masks):
    pred, gt = known_masks
    assert iou_score(pred, gt) == pytest.approx(2 / 6)


def test_perfect_match_gives_score_of_one():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    assert precision_score(mask, mask) == 1.0
    assert recall_score(mask, mask) == 1.0
    assert dice_score(mask, mask) == 1.0
    assert iou_score(mask, mask) == 1.0


def test_completely_disjoint_masks_give_score_of_zero():
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[0, 0] = 1
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[3, 3] = 1
    assert precision_score(pred, gt) == 0.0
    assert recall_score(pred, gt) == 0.0
    assert dice_score(pred, gt) == 0.0
    assert iou_score(pred, gt) == 0.0


def test_both_empty_masks_quirk_iou_one_but_dice_zero():
    """
    Kuirk yang layak diketahui pengembang: kalau prediksi DAN ground truth
    sama-sama kosong (tidak ada piksel positif sama sekali), iou_score()
    mengembalikan 1.0 (union=0 -> dianggap 'sepakat sempurna'), TAPI
    precision_score()/recall_score()/dice_score() mengembalikan 0.0
    (karena predikat `(tp + fp) > 0` bernilai False, bukan dianggap
    'undefined -> 1.0' seperti IoU). Ini bukan bug yang perlu diperbaiki
    di sini, tapi INKONSISTENSI DEFINISI yang layak diketahui siapa pun
    yang membaca metrik dari citra kosong (mis. background-only patch).
    """
    empty = np.zeros((4, 4), dtype=np.uint8)
    assert iou_score(empty, empty) == 1.0
    assert precision_score(empty, empty) == 0.0
    assert recall_score(empty, empty) == 0.0
    assert dice_score(empty, empty) == 0.0


def test_confluence_percent_matches_hand_calculation(known_masks):
    pred, gt = known_masks
    # 4 piksel foreground dari total 16 piksel = 25%
    assert confluence_percent(pred) == pytest.approx(25.0)
    assert confluence_percent(gt) == pytest.approx(25.0)


def test_confluence_percent_of_empty_array_is_zero():
    assert confluence_percent(np.zeros((0, 0), dtype=np.uint8)) == 0.0


def test_percentage_error_zero_when_confluence_matches():
    assert percentage_error(pred_conf=25.0, actual_conf=25.0) == 0.0


def test_percentage_error_matches_hand_calculation():
    # |30 - 20| / 20 * 100 = 50%
    assert percentage_error(pred_conf=30.0, actual_conf=20.0) == pytest.approx(50.0)


def test_compute_all_metrics_without_ground_truth_returns_none_fields():
    """FR05: inferensi tanpa label -> FR06 tidak boleh memaksakan angka
    perbandingan yang sebenarnya tidak ada datanya."""
    pred = np.ones((4, 4), dtype=np.uint8)
    result = compute_all_metrics(pred, gt=None, t=0.123)

    assert result["iou"] is None
    assert result["dice"] is None
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["confluence_actual"] is None
    assert result["percentage_error"] is None
    # Yang TIDAK butuh ground truth tetap harus terisi:
    assert result["confluence_pred"] == pytest.approx(100.0)
    assert result["inference_time_sec"] == pytest.approx(0.123)


def test_compute_all_metrics_with_ground_truth_fills_all_fields(known_masks):
    pred, gt = known_masks
    result = compute_all_metrics(pred, gt=gt, t=0.5)

    assert result["iou"] == pytest.approx(2 / 6)
    assert result["dice"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["confluence_pred"] == pytest.approx(25.0)
    assert result["confluence_actual"] == pytest.approx(25.0)
    assert result["percentage_error"] == pytest.approx(0.0)


def test_compute_all_metrics_raises_on_shape_mismatch():
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_all_metrics(pred, gt=gt, t=0.0)


def test_average_metrics_computes_mean_and_renames_dice_key():
    mlist = [
        {"iou": 0.4, "dice": 0.5, "percentage_error": 10.0, "inference_time_sec": 1.0},
        {"iou": 0.6, "dice": 0.7, "percentage_error": 20.0, "inference_time_sec": 3.0},
    ]
    avg = average_metrics(mlist)

    assert avg["iou"] == pytest.approx(0.5)
    assert avg["dice_coefficient"] == pytest.approx(0.6)  # key di-rename dari 'dice'
    assert avg["percentage_error"] == pytest.approx(15.0)
    assert avg["inference_time_sec"] == pytest.approx(2.0)


def test_average_metrics_ignores_none_values():
    """Kalau sebagian citra dalam batch tidak punya label (iou=None),
    rata-rata seharusnya cuma dihitung dari yang punya nilai, bukan
    dianggap 0 (yang akan bias hasil rata-rata ke bawah)."""
    mlist = [
        {"iou": 0.8, "dice": 0.9, "percentage_error": 5.0, "inference_time_sec": 1.0},
        {"iou": None, "dice": None, "percentage_error": None, "inference_time_sec": 2.0},
    ]
    avg = average_metrics(mlist)

    assert avg["iou"] == pytest.approx(0.8)  # bukan (0.8+0)/2 = 0.4
    assert avg["inference_time_sec"] == pytest.approx(1.5)  # ini memang selalu ada, jadi dirata-rata semua


def test_average_metrics_all_none_returns_none():
    mlist = [{"iou": None, "dice": None, "percentage_error": None, "inference_time_sec": 1.0}]
    avg = average_metrics(mlist)
    assert avg["iou"] is None
    assert avg["dice_coefficient"] is None
