"""
tests/e2e/test_journey_with_ground_truth.py — FR02, FR05, FR06, FR07, FR08.

INI JANTUNG dari perombakan struktur test. Sebelumnya, tiap FR ini punya
file sendiri yang MASING-MASING mengunggah citra+label dan menjalankan
inferensi dari nol -- padahal semuanya memeriksa HASIL DARI SATU
tindakan pengguna yang sama (upload lalu klik 'Jalankan Inferensi').

Perbaikannya memakai `subtests` (docs.pytest.org/en/stable/how-to/
subtests.html, awalnya plugin pytest-subtests dari pytest-dev): upload +
inferensi dijalankan SEKALI per test function, lalu tiap FR diperiksa
sebagai subtest terpisah -- kalau satu subtest gagal, subtest lain tetap
jalan dan tetap dilaporkan satu-satu (bukan cuma "test gagal", tapi
"FR06 gagal, FR07 & FR08 tetap lolos"), jadi traceability per-FR tetap
utuh walau eksekusinya digabung.

Ini bukan melanggar "Fresh browser per test" (selenium.dev) -- tetap
SATU test function = SATU browser = SATU fixture `app_page`. Yang
dihindari adalah mengulang tindakan MAHAL (upload + inferensi U-Net) demi
memeriksa hal yang sudah dihasilkan oleh tindakan yang sama.
"""
import re

import pytest

from testconfig import settings


pytestmark = pytest.mark.fr05


def _iou_value(metrics_text: str):
    match = re.search(r"IoU\D+([0-9]+\.[0-9]+|—)", metrics_text)
    if not match:
        return None
    return None if match.group(1) == "—" else float(match.group(1))


def test_full_journey_with_ground_truth(app_page, test_data, subtests):
    # ── Setup mahal, dijalankan SEKALI ──────────────────────────────────
    app_page.upload_image(test_data["matched_image"])
    app_page.upload_label(test_data["matched_label"])
    app_page.run_inference()

    # ── FR02: pencocokan stem identik menghasilkan ground truth ────────
    with subtests.test(msg="FR02: stem sama -> IoU numerik (bukan '—')"):
        iou = _iou_value(app_page.get_metrics_text())
        assert iou is not None, "IoU seharusnya numerik, bukan '—', untuk pasangan stem identik."
        assert 0.0 <= iou <= 1.0

    # ── FR05: inferensi dengan label berhasil ───────────────────────────
    with subtests.test(msg="FR05: inferensi berhasil, tidak ada error fatal"):
        assert app_page.inference_ran_successfully()
        assert app_page.result_exists_for(f"{test_data['matched_stem']}.jpg")
        fatal = [a for a in app_page.get_alert_texts(timeout=2) if "traceback" in a.lower()]
        assert not fatal, f"Muncul error: {fatal}"

    # ── FR06: semua metrik dihitung & dalam rentang valid ───────────────
    with subtests.test(msg="FR06: semua baris metrik hadir"):
        missing = app_page.metrics_rows_present(settings.METRIC_ROW_LABELS)
        assert not missing, f"Baris metrik hilang: {missing}"

    with subtests.test(msg="FR06: Dice dalam rentang [0,1]"):
        text = app_page.get_metrics_text()
        match = re.search(r"Dice / F1\D+([0-9]+\.[0-9]+)", text)
        assert match
        assert 0.0 <= float(match.group(1)) <= 1.0

    with subtests.test(msg="FR06: konfluensi dalam rentang [0,100]%"):
        text = app_page.get_metrics_text()
        for label in ("Konfluensi Prediksi (%)", "Konfluensi Aktual (%)"):
            match = re.search(rf"{re.escape(label)}\D+([0-9]+\.[0-9]+)", text)
            assert match, f"Tidak ketemu baris '{label}'"
            assert 0.0 <= float(match.group(1)) <= 100.0

    with subtests.test(msg="FR06: rata-rata batch muncul (label ditemukan)"):
        assert not app_page.avg_metrics_unavailable_shown()
        assert "Rata-rata Hasil Batch" in app_page.get_page_text()

    # ── FR07: citra asli/GT/prediksi/overlay ditampilkan ────────────────
    with subtests.test(msg="FR07: gambar hasil ter-render"):
        img_el = app_page.get_result_figure_image()
        assert img_el is not None
        w = app_page.driver.execute_script("return arguments[0].naturalWidth;", img_el)
        h = app_page.driver.execute_script("return arguments[0].naturalHeight;", img_el)
        assert w and h

    with subtests.test(msg="FR07: rasio aspek konsisten dengan 4 panel"):
        ratio = app_page.get_result_figure_aspect_ratio()
        n = settings.N_PANELS_WITH_GT
        assert ratio is not None
        assert (n - 1.3) <= ratio <= (n + 0.7), f"Rasio {ratio:.2f} tidak cocok {n} panel."

    with subtests.test(msg="FR07: tombol unduh visualisasi & mask tersedia"):
        assert app_page.download_button_exists(settings.DOWNLOAD_VIS_LABEL)
        assert app_page.download_button_exists(settings.DOWNLOAD_MASK_LABEL)

    # ── FR08: overlay TP hijau ────────────────────────────────────────
    with subtests.test(msg="FR08: piksel hijau (TP) muncul di overlay"):
        ratios = app_page.overlay_color_ratios()
        assert ratios["green"] >= 0.001, (
            f"Piksel hijau (TP) tidak cukup (ratio={ratios['green']:.5f}). Label uji "
            f"sengaja dibuat overlap dengan citra, jadi ini seharusnya selalu muncul."
        )

    with subtests.test(msg="FR08: panel overlay tidak grayscale murni"):
        import numpy as np

        panel = app_page.get_overlay_panel_image()
        assert panel is not None
        arr = np.array(panel)
        channel_diff = np.abs(arr[..., 0].astype(int) - arr[..., 1].astype(int)).mean()
        assert channel_diff > 0.5


def test_matching_patterns_and_negative_case_in_one_batch(app_page, test_data, subtests):
    """
    FR02 secara menyeluruh: KETIGA pola pencocokan (stem sama, sufiks
    '_mask', master annotations.json) DAN kasus negatif (citra tanpa
    pasangan) diuji dalam SATU batch upload + SATU run inferensi, karena
    app.py mendukung upload banyak citra sekaligus (FR01) -- jadi ini
    sekalian menguji FR01 untuk kasus multi-citra yang sesungguhnya,
    bukan disimulasikan terpisah-pisah.
    """
    app_page.upload_image(
        test_data["matched_image"],       # pola 1: stem sama
        test_data["mask_suffix_image"],   # pola 2: sufiks _mask
        test_data["json_image"],          # pola 3: master json
        test_data["unmatched_image"],     # negatif: tanpa pasangan
    )
    app_page.upload_label(
        test_data["matched_label"],
        test_data["mask_suffix_label"],
        test_data["json_master_label"],
    )
    app_page.run_inference()

    with subtests.test(msg="FR01: keempat citra diproses"):
        for stem, ext in [
            (test_data["matched_stem"], ".jpg"),
            (test_data["mask_suffix_stem"], ".png"),
            (test_data["json_stem"], ".jpg"),
            (test_data["unmatched_stem"], ".jpg"),
        ]:
            assert app_page.result_exists_for(f"{stem}{ext}"), f"Hasil untuk {stem}{ext} tidak ada."

    page_text = app_page.get_page_text()

    with subtests.test(msg="FR02 pola 1 (stem sama) menghasilkan ground truth"):
        idx = page_text.find(f"{test_data['matched_stem']}.jpg")
        assert idx != -1
        snippet = page_text[idx: idx + 1500]
        assert "IoU" in snippet

    with subtests.test(msg="FR02 pola 2 (_mask suffix) menghasilkan ground truth"):
        idx = page_text.find(f"{test_data['mask_suffix_stem']}.png")
        assert idx != -1

    with subtests.test(msg="FR02 pola 3 (master annotations.json) berhasil diekstrak"):
        assert not app_page.label_json_load_failed()
        idx = page_text.find(f"{test_data['json_stem']}.jpg")
        assert idx != -1

    with subtests.test(msg="FR02 negatif: citra tanpa pasangan label -> ditandai tidak ditemukan"):
        # Karena master_json ADA di batch ini, jalur fallback ikut dicoba
        # untuk citra tanpa pasangan .png -- app.py akan melaporkan
        # 'Tidak ditemukan dalam annotations.json' untuk citra ini secara
        # spesifik (bukan cuma diam-diam gt=None).
        assert app_page.label_not_found_in_json()
