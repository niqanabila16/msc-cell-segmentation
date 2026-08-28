"""
tests/e2e/test_upload_and_validation.py — FR01, FR03, FR04.

Semua test di sini BERHENTI DI TAHAP UPLOAD (tidak menjalankan inferensi
U-Net), jadi tetap murah walau tiap test tetap dapat browser baru sendiri
("Fresh browser per test" -- selenium.dev/documentation/test_practices/
encouraged/fresh_browser_per_test/). Sengaja TIDAK digabung jadi satu
test besar pakai subtests: upload sendiri murah (di bawah 1 detik), jadi
tidak ada "setup mahal" yang perlu dibagi -- justru kalau dipaksa
digabung dalam SATU sesi upload, ada risiko file dari skenario
sebelumnya masih "menempel" di widget (Streamlit uploader bersifat
akumulatif kalau tidak dihapus manual), yang bisa mencemari skenario
sesudahnya. Subtests baru dipakai di test_journey_*.py, karena di sana
setup yang mahal (inferensi U-Net) memang perlu dibagi.

FR03/FR04 di sini menguji PERILAKU APLIKASI YANG SEDANG BERJALAN (beda
level dengan tests/unit/test_data_validation.py yang menguji fungsi
validate_image_file()/validate_label_file() itu sendiri, terlepas dari
apakah app.py memanggilnya). Temuan: app.py TIDAK memanggil kedua fungsi
itu -- penegakan FR03/FR04 di app.py yang berjalan sepenuhnya bergantung
pada `type=` bawaan st.file_uploader() (format) dan `server.maxUploadSize`
di .streamlit/config.toml (ukuran). Lihat README.md untuk detail.
"""
from selenium.webdriver.common.by import By

import pytest

from testconfig import settings


# ── FR01 ──────────────────────────────────────────────────────────────

@pytest.mark.fr01
def test_uploader_widgets_are_present(app_page):
    uploaders = app_page.driver.find_elements(By.CSS_SELECTOR, '[data-testid="stFileUploader"]')
    assert len(uploaders) >= 2
    app_page.find_widget_by_label(settings.IMAGE_UPLOADER_LABEL, "stFileUploader")
    app_page.find_widget_by_label(settings.LABEL_UPLOADER_LABEL, "stFileUploader")


@pytest.mark.fr01
def test_multiple_images_can_be_selected_at_once(app_page, test_data):
    """FR01: 'satu atau lebih citra ... melalui antarmuka'."""
    app_page.upload_image(test_data["matched_image"], test_data["mask_suffix_image"])
    app_page.upload_label(test_data["matched_label"], test_data["mask_suffix_label"])

    assert app_page.image_uploader_shows_filename(f"{test_data['matched_stem']}.jpg")
    assert app_page.image_uploader_shows_filename(f"{test_data['mask_suffix_stem']}.png")


@pytest.mark.fr01
def test_image_without_label_does_not_block_upload(app_page, test_data):
    """FR01 menyebut label sebagai pelengkap ('beserta label'); tidak
    boleh jadi syarat wajib di tahap upload (FR05 membolehkan tanpa label)."""
    app_page.upload_image(test_data["unmatched_image"])
    blocking = [a for a in app_page.get_alert_texts(timeout=2) if "wajib" in a.lower()]
    assert not blocking


# ── FR03: format & ukuran citra ─────────────────────────────────────────

@pytest.mark.fr03
def test_valid_image_formats_are_accepted(app_page, test_data):
    app_page.upload_image(test_data["matched_image"])  # .jpg
    assert app_page.image_uploader_shows_filename(f"{test_data['matched_stem']}.jpg")


@pytest.mark.fr03
def test_unsupported_image_format_is_rejected_at_widget_level(app_page, test_data):
    app_page.upload_image(test_data["invalid_format_image"])
    app_page.click_button(settings.RUN_INFERENCE_BUTTON_TEXT)
    assert app_page.no_images_error_shown(), (
        f"Upload HANYA citra .bmp lalu klik run seharusnya memicu "
        f"'{settings.MSG_NO_IMAGES}' karena Streamlit menolaknya di level "
        f"widget (type=[\"jpg\",\"jpeg\",\"png\"])."
    )


@pytest.mark.fr03
def test_oversized_image_is_rejected(app_page, test_data):
    app_page.upload_image(test_data["oversized_image"])
    app_page.click_button(settings.RUN_INFERENCE_BUTTON_TEXT)
    assert app_page.no_images_error_shown(), (
        f"Kalau test ini GAGAL: kemungkinan besar `server.maxUploadSize` "
        f"belum diset ke {settings.MAX_FILE_SIZE_MB} di .streamlit/config.toml "
        f"-- lihat README.md bagian temuan, ini bukan berarti test salah tulis."
    )


# ── FR04: format & ukuran label ─────────────────────────────────────────

@pytest.mark.fr04
def test_valid_label_format_is_accepted(app_page, test_data):
    app_page.upload_label(test_data["matched_label"])
    assert app_page.label_uploader_shows_filename(f"{test_data['matched_stem']}.png")


@pytest.mark.fr04
def test_unsupported_label_format_is_rejected_at_widget_level(app_page, test_data):
    app_page.upload_label(test_data["invalid_format_label"])
    filename_only = test_data["invalid_format_label"].split("/")[-1].split("\\")[-1]
    assert not app_page.label_uploader_shows_filename(filename_only)


@pytest.mark.fr04
def test_oversized_label_is_rejected(app_page, test_data):
    app_page.upload_label(test_data["oversized_label"])
    assert not app_page.label_uploader_shows_filename("oversized_label.png"), (
        "Kalau test ini GAGAL (file tetap tercantum): kemungkinan "
        "server.maxUploadSize belum diset -- lihat README.md."
    )


@pytest.mark.fr04
def test_corrupt_png_label_is_reported_during_inference(app_page, test_data):
    """Satu-satunya test FR04 yang perlu menjalankan inferensi (karena
    pesan 'Gagal memuat mask PNG' baru muncul saat _parse_uploads()
    dieksekusi di app.py, yaitu setelah tombol run diklik)."""
    app_page.upload_image(test_data["matched_image"])
    app_page.upload_label(test_data["corrupt_label_matched_name"])
    app_page.run_inference()

    assert app_page.label_png_load_failed(), (
        f"Label .png korup dengan nama cocok seharusnya memicu peringatan "
        f"yang memuat '{settings.MSG_LABEL_PNG_LOAD_FAIL_SUBSTR}'."
    )
    assert app_page.result_exists_for(f"{test_data['matched_stem']}.jpg"), (
        "Citra tetap harus diproses (tanpa ground truth) walau labelnya korup."
    )
