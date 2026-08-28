"""
testconfig/settings.py — Konfigurasi terpusat untuk automation testing Selenium.

Semua teks di bawah ini (kecuali yang ditandai "ASUMSI") diambil PERSIS
dari app.py yang Anda lampirkan. Kalau UI berubah, cukup edit di sini.

Nama folder ini sengaja "testconfig", BUKAN "config" — supaya tidak
tabrakan dengan config.py di root proyek (ape_msc/config.py) saat pytest
menambahkan src/test/ ke sys.path. `import config` di dalam proses
pytest kalau folder ini bernama "config" bisa saja secara tidak sengaja
me-resolve ke package test ini alih-alih config.py proyek, tergantung
urutan sys.path — nama berbeda menghilangkan risiko itu sepenuhnya.
"""
import os

# ─────────────────────────────────────────────────────────────────────────
# Koneksi & environment
# ─────────────────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("APE_MSC_BASE_URL", "http://localhost:8501")

DEFAULT_TIMEOUT = 15
STREAMLIT_RERUN_TIMEOUT = 12
INFERENCE_TIMEOUT = 90            # inferensi U-Net di CPU bisa lambat

BROWSER = os.environ.get("APE_MSC_BROWSER", "chrome")   # "chrome" | "firefox"
HEADLESS = os.environ.get("APE_MSC_HEADLESS", "true").lower() == "true"
WINDOW_SIZE = (1600, 1000)

# ─────────────────────────────────────────────────────────────────────────
# FR03 / FR04 — ekstensi & ukuran
# ─────────────────────────────────────────────────────────────────────────
# TERVERIFIKASI dari app.py: st.file_uploader(..., type=["jpg","jpeg","png"])
ALLOWED_IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

# TERVERIFIKASI dari app.py: st.file_uploader(..., type=["png","json"])
# CATATAN PENTING: FR04 di tabel kebutuhan fungsional hanya menyebut
# ".png", tapi implementasi aktual JUGA menerima ".json" (master COCO
# annotations, lihat _parse_uploads()). Ini perlu diselaraskan di
# dokumen requirement, atau memang .json harus dianggap bagian dari FR04.
ALLOWED_LABEL_EXTS = [".png", ".json"]

MAX_FILE_SIZE_MB = 10
# ASUMSI: sidebar app.py menyatakan "Batas upload: 10 MB / file", tapi
# validate_image_file()/validate_label_file() dari src/data_utils.py
# (yang berisi logika pengecekan 10MB & pesan "melebihi batas") DI-IMPORT
# di app.py TAPI TIDAK PERNAH DIPANGGIL di _parse_uploads() maupun di
# tempat lain yang terlihat. Artinya batas 10MB kemungkinan besar HANYA
# ditegakkan lewat setting `server.maxUploadSize` di .streamlit/config.toml
# (kalau ada), bukan lewat validasi custom aplikasi. Cek file itu di
# proyek Anda. Test FR03/FR04 terkait ukuran file akan gagal dengan pesan
# yang menjelaskan hal ini kalau asumsi ini keliru di environment Anda.
ASSUMED_SERVER_MAX_UPLOAD_MB = 10

# ─────────────────────────────────────────────────────────────────────────
# Label widget — TERVERIFIKASI dari app.py
# ─────────────────────────────────────────────────────────────────────────
IMAGE_UPLOADER_LABEL = "Pilih satu atau lebih file citra"
LABEL_UPLOADER_LABEL = "Pilih file label (.png / .json)"
VERSION_SELECT_LABEL = "Versi / Konfigurasi"
RUN_INFERENCE_BUTTON_TEXT = "Jalankan Inferensi"

# TERVERIFIKASI dari src/ui_helpers.py (dipanggil apa adanya oleh app.py)
SLIDER_ALPHA_LABEL = "Transparansi Overlay"
SLIDER_ALPHA_MIN = 0.1
SLIDER_ALPHA_MAX = 0.9
SLIDER_ALPHA_STEP = 0.05

SELECTBOX_PREPROCESSING_LABEL = "preprocessing Citra"
PREPROCESSING_OPTIONS = [
    "Tanpa Preprocessing",
    "Polynomial Background Correction",
    "Percentile Normalization",
    "Polynomial + Percentile",
]

# TERVERIFIKASI: download_png() dipanggil TANPA override label di app.py
# -> memakai default dari ui_helpers.py
DOWNLOAD_VIS_LABEL = "Unduh Visualisasi"

# TERVERIFIKASI dari app.py: download_mask_png(..., label="Unduh Mask Hitam-Putih", ...)
# -> app.py MENG-OVERRIDE default ui_helpers.py ("Unduh Mask"). Pakai teks ini.
DOWNLOAD_MASK_LABEL = "Unduh Mask Hitam-Putih"

# TERVERIFIKASI dari app.py: download_csv(..., "Unduh Semua Hasil (CSV)")
# -> hanya muncul kalau > 1 citra diproses sekaligus (len(all_results) > 1)
DOWNLOAD_CSV_LABEL = "Unduh Semua Hasil (CSV)"

# TERVERIFIKASI dari src/ui_helpers.py::metrics_to_df (tidak diubah app.py)
METRIC_ROW_LABELS = [
    "IoU",
    "Dice / F1",
    "Precision",
    "Recall",
    "Konfluensi Prediksi (%)",
    "Konfluensi Aktual (%)",
    "Waktu Inferensi",
]

# TERVERIFIKASI dari src/visualization.py (tidak diubah app.py)
N_PANELS_WITH_GT = 4       # Citra Asli, Ground Truth, Prediksi, Overlay
N_PANELS_WITHOUT_GT = 3    # Citra Asli, Prediksi, Overlay

# ─────────────────────────────────────────────────────────────────────────
# Pesan & teks — TERVERIFIKASI persis dari app.py (string literal di kode)
# ─────────────────────────────────────────────────────────────────────────
MSG_NO_IMAGES = "Belum ada citra yang di-upload."                 # imgs_up kosong saat klik run
MSG_NO_VALID_IMAGES = "Tidak ada citra valid."                    # semua citra gagal dimuat
MSG_INFERENCE_SUCCESS_SUBSTR = "citra berhasil diproses."         # show_success(...)
MSG_AVG_METRICS_UNAVAILABLE_SUBSTR = "Label tidak ditemukan"      # avg_ph.info(...) saat semua gt None
MSG_LABEL_PNG_LOAD_FAIL_SUBSTR = "Gagal memuat mask PNG"          # label .png korup / gagal dibaca
MSG_LABEL_JSON_LOAD_FAIL_SUBSTR = "Gagal ekstrak dari annotations.json"
MSG_LABEL_NOT_IN_JSON_SUBSTR = "Tidak ditemukan dalam annotations.json"
MSG_IMAGE_LOAD_FAIL_SUBSTR = "Gagal memuat citra"

# ─────────────────────────────────────────────────────────────────────────
# Path
# ─────────────────────────────────────────────────────────────────────────
TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/test
TEST_DATA_DIR = os.path.join(TEST_ROOT, "fixtures", "test_data")
SCREENSHOT_DIR = os.path.join(TEST_ROOT, "screenshots")
