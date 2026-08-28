"""
pages/app_page.py — Page object khusus aplikasi ape_msc, dibangun di atas
BasePage dengan method bermakna per kebutuhan fungsional (FR). Semua
teks/pesan yang dicek di sini bersumber dari testconfig.settings, yang
sudah diverifikasi terhadap app.py.
"""
from typing import List, Optional

import numpy as np
from PIL import Image

from testconfig import settings
from pages.base_page import BasePage


class AppPage(BasePage):
    def open(self):
        self.driver.get(settings.BASE_URL)
        self.wait_for_app_ready()
        return self

    # ── FR01: upload citra & label ──────────────────────────────────────

    def upload_image(self, *file_paths: str):
        self.upload_files(settings.IMAGE_UPLOADER_LABEL, list(file_paths))
        return self

    def upload_label(self, *file_paths: str):
        self.upload_files(settings.LABEL_UPLOADER_LABEL, list(file_paths))
        return self

    def image_uploader_shows_filename(self, filename: str) -> bool:
        """
        True kalau nama file MASIH tercantum sebagai file yang diterima
        di dalam widget uploader (dipakai untuk mendeteksi apakah
        Streamlit menolak file di level widget -- lihat FR03/FR04).
        """
        text = self.get_uploader_container_text(settings.IMAGE_UPLOADER_LABEL)
        return filename in text

    def label_uploader_shows_filename(self, filename: str) -> bool:
        text = self.get_uploader_container_text(settings.LABEL_UPLOADER_LABEL)
        return filename in text

    # ── FR05: jalankan inferensi ─────────────────────────────────────────

    def run_inference(self):
        self.click_button(settings.RUN_INFERENCE_BUTTON_TEXT)
        self.wait_for_spinner_gone()     # st.spinner saat memuat model
        self.wait_for_progress_gone()    # st.progress saat loop per-citra
        self.wait_for_rerun_complete()
        return self

    # ── Deteksi pesan yang TERVERIFIKASI persis dari app.py ─────────────

    def has_alert_containing(self, substr: str, timeout: int = 3) -> bool:
        alerts = self.get_alert_texts(timeout=timeout)
        return any(substr in a for a in alerts)

    def no_images_error_shown(self) -> bool:
        return self.has_alert_containing(settings.MSG_NO_IMAGES)

    def no_valid_images_error_shown(self) -> bool:
        return self.has_alert_containing(settings.MSG_NO_VALID_IMAGES)

    def inference_success_shown(self) -> bool:
        return self.has_alert_containing(settings.MSG_INFERENCE_SUCCESS_SUBSTR)

    def avg_metrics_unavailable_shown(self) -> bool:
        return self.has_alert_containing(settings.MSG_AVG_METRICS_UNAVAILABLE_SUBSTR)

    def label_png_load_failed(self) -> bool:
        return self.has_alert_containing(settings.MSG_LABEL_PNG_LOAD_FAIL_SUBSTR)

    def label_json_load_failed(self) -> bool:
        return self.has_alert_containing(settings.MSG_LABEL_JSON_LOAD_FAIL_SUBSTR)

    def label_not_found_in_json(self) -> bool:
        return self.has_alert_containing(settings.MSG_LABEL_NOT_IN_JSON_SUBSTR)

    # ── FR01/FR02/FR05: hasil per-citra (dibungkus st.expander(rec['name'])) ─

    def get_result_titles(self) -> List[str]:
        """
        Judul tiap expander hasil = nama file citra (lihat app.py:
        `with st.expander(f"{rec['name']}", expanded=True):`). Baris
        pertama dari teks expander adalah judulnya.
        """
        return [t.split("\n")[0].strip() for t in self.get_expander_texts() if t.strip()]

    def result_exists_for(self, image_filename: str) -> bool:
        return any(image_filename in title for title in self.get_result_titles())

    def inference_ran_successfully(self) -> bool:
        return self.inference_success_shown() or len(self.get_result_titles()) > 0

    # ── FR06: metrik evaluasi ────────────────────────────────────────────

    def get_metrics_text(self) -> str:
        table_text = self.get_dataframe_text()
        return table_text if table_text.strip() else self.get_page_text()

    def metrics_rows_present(self, expected_rows: List[str] = None) -> List[str]:
        expected_rows = expected_rows or settings.METRIC_ROW_LABELS
        text = self.get_metrics_text()
        return [row for row in expected_rows if row not in text]

    # ── FR07: figure gabungan (Citra Asli / GT / Prediksi / Overlay) ────

    def get_result_figure_image(self):
        images = self.get_all_image_elements()
        if not images:
            return None
        return max(images, key=lambda el: el.size.get("width", 0))

    def get_result_figure_aspect_ratio(self) -> Optional[float]:
        img_el = self.get_result_figure_image()
        if img_el is None:
            return None
        pil_img = self.get_image_pil(img_el)
        if pil_img is None:
            return None
        w, h = pil_img.size
        return w / h if h else None

    # ── FR08: warna overlay TP/FP/FN ─────────────────────────────────────

    def get_overlay_panel_image(self, n_panels: int = None) -> Optional[Image.Image]:
        n_panels = n_panels or settings.N_PANELS_WITH_GT
        img_el = self.get_result_figure_image()
        if img_el is None:
            return None
        pil_img = self.get_image_pil(img_el)
        if pil_img is None:
            return None
        w, h = pil_img.size
        panel_w = w // n_panels
        return pil_img.crop((w - panel_w, 0, w, h))

    @staticmethod
    def _hue_pixel_ratio(pil_img: Image.Image, hue_check) -> float:
        arr = np.array(pil_img.convert("RGB"))
        r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
        mask = hue_check(r, g, b)
        return float(mask.sum()) / float(mask.size)

    def overlay_color_ratios(self, n_panels: int = None) -> dict:
        panel = self.get_overlay_panel_image(n_panels)
        if panel is None:
            return {"green": 0.0, "yellow": 0.0, "red": 0.0}
        green = self._hue_pixel_ratio(panel, lambda r, g, b: (g > r + 25) & (g > b + 25))
        yellow = self._hue_pixel_ratio(
            panel, lambda r, g, b: (r > 120) & (g > 120) & (b < r - 40) & (b < g - 40)
        )
        red = self._hue_pixel_ratio(panel, lambda r, g, b: (r > g + 25) & (r > b + 25))
        return {"green": green, "yellow": yellow, "red": red}
