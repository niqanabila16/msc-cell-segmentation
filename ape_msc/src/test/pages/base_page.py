"""
pages/base_page.py — Helper generik untuk berinteraksi dengan komponen
Streamlit lewat Selenium.

Dipakai bersama oleh semua page object. Mencari widget berdasarkan teks
labelnya (bukan `key=` internal) supaya tidak bergantung pada detail
implementasi yang tidak tercermin di DOM.
"""
import base64
import io
import os
import time
from typing import List, Optional

from PIL import Image
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from testconfig import settings


class BasePage:
    def __init__(self, driver):
        self.driver = driver

    # ── Wait helpers ───────────────────────────────────────────────────

    def wait(self, timeout: int = None) -> WebDriverWait:
        return WebDriverWait(self.driver, timeout or settings.DEFAULT_TIMEOUT)

    def wait_for_app_ready(self, timeout: int = None) -> None:
        self.wait(timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stApp"]'))
        )
        self.wait_for_rerun_complete(timeout)

    def wait_for_rerun_complete(self, timeout: int = None) -> None:
        """Tunggu indikator 'running' Streamlit (kanan atas) hilang setelah interaksi."""
        timeout = timeout or settings.STREAMLIT_RERUN_TIMEOUT
        try:
            time.sleep(0.3)
            self.wait(timeout).until_not(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, '[data-testid="stStatusWidget"]')
                )
            )
        except TimeoutException:
            pass

    def wait_for_spinner_gone(self, timeout: int = None) -> None:
        timeout = timeout or settings.INFERENCE_TIMEOUT
        try:
            self.wait(2).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stSpinner"]'))
            )
        except TimeoutException:
            pass
        self.wait(timeout).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stSpinner"]'))
        )

    def wait_for_progress_gone(self, timeout: int = None) -> None:
        """app.py menampilkan st.progress() selama loop inferensi per-citra."""
        timeout = timeout or settings.INFERENCE_TIMEOUT
        try:
            self.wait(timeout).until_not(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stProgress"]'))
            )
        except TimeoutException:
            pass

    # ── Locator umum berbasis teks label ──────────────────────────────

    def _widget_block_by_label_xpath(self, label_text: str, testid: str) -> str:
        return (
            f'//div[@data-testid="{testid}"]'
            f'[.//*[normalize-space(text())="{label_text}"]]'
        )

    def find_widget_by_label(self, label_text: str, testid: str, timeout: int = None):
        xpath = self._widget_block_by_label_xpath(label_text, testid)
        return self.wait(timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))

    # ── File uploader ──────────────────────────────────────────────────

    def upload_files(self, label_text: str, file_paths: List[str], timeout: int = None) -> None:
        container = self.find_widget_by_label(label_text, "stFileUploader", timeout)
        file_input = container.find_element(By.CSS_SELECTOR, 'input[type="file"]')
        abs_paths = "\n".join(os.path.abspath(p) for p in file_paths)
        try:
            file_input.send_keys(abs_paths)
        except Exception:
            self.driver.execute_script(
                "arguments[0].style.display='block'; arguments[0].style.opacity=1;",
                file_input,
            )
            file_input.send_keys(abs_paths)
        self.wait_for_rerun_complete(timeout)

    def get_uploader_container_text(self, label_text: str, testid: str = "stFileUploader",
                                     timeout: int = 5) -> str:
        try:
            container = self.find_widget_by_label(label_text, testid, timeout)
            return container.text
        except TimeoutException:
            return ""

    # ── Selectbox (dropdown) ────────────────────────────────────────────

    def select_dropdown_option(self, label_text: str, option_text: str, timeout: int = None) -> None:
        container = self.find_widget_by_label(label_text, "stSelectbox", timeout)
        select_box = container.find_element(By.CSS_SELECTOR, '[data-baseweb="select"]')
        select_box.click()

        listbox = self.wait(timeout).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'ul[role="listbox"]'))
        )
        options = listbox.find_elements(By.CSS_SELECTOR, 'li, [role="option"]')
        for opt in options:
            if opt.text.strip() == option_text.strip():
                opt.click()
                self.wait_for_rerun_complete(timeout)
                return
        raise NoSuchElementException(
            f"Opsi '{option_text}' tidak ditemukan pada dropdown '{label_text}'. "
            f"Opsi yang tersedia: {[o.text for o in options]}"
        )

    def get_selected_dropdown_value(self, label_text: str, timeout: int = None) -> str:
        container = self.find_widget_by_label(label_text, "stSelectbox", timeout)
        select_box = container.find_element(By.CSS_SELECTOR, '[data-baseweb="select"]')
        return select_box.text.strip()

    # ── Slider ───────────────────────────────────────────────────────────

    def set_slider_value(
        self, label_text: str, target_value: float,
        min_value: float, max_value: float, step: float, timeout: int = None,
    ) -> float:
        """
        PERBAIKAN: sebelumnya memakai Keys.HOME untuk reset ke nilai
        minimum sebelum menekan ARROW_RIGHT sejumlah step -- ternyata
        Keys.HOME tidak reliable mereset slider BaseWeb (dikonfirmasi dari
        hasil test: hanya target=nilai MAKSIMUM yang lolos, karena nilai
        itu tetap tercapai lewat clamping berapa pun titik awalnya, target
        lain meleset). Sebagai gantinya, tekan ARROW_LEFT lebih banyak
        dari mungkin diperlukan (dijamin mentok ke minimum lewat clamping
        browser, terlepas dari nilai awal), baru tekan ARROW_RIGHT
        sejumlah step yang dibutuhkan -- deterministik tanpa bergantung
        pada dukungan tombol Home.
        """
        container = self.find_widget_by_label(label_text, "stSlider", timeout)
        handle = container.find_element(By.CSS_SELECTOR, '[role="slider"]')

        ActionChains(self.driver).move_to_element(handle).click().perform()
        time.sleep(0.1)

        max_possible_steps = int(round((max_value - min_value) / step)) + 3
        for _ in range(max_possible_steps):
            handle.send_keys(Keys.ARROW_LEFT)  # dijamin mentok ke minimum

        n_steps = int(round((target_value - min_value) / step))
        for _ in range(max(n_steps, 0)):
            handle.send_keys(Keys.ARROW_RIGHT)
        self.wait_for_rerun_complete(timeout)

        current = handle.get_attribute("aria-valuenow")
        return float(current) if current is not None else None

    def get_slider_value(self, label_text: str, timeout: int = None) -> float:
        container = self.find_widget_by_label(label_text, "stSlider", timeout)
        handle = container.find_element(By.CSS_SELECTOR, '[role="slider"]')
        return float(handle.get_attribute("aria-valuenow"))

    # ── Tombol ───────────────────────────────────────────────────────────

    # PENTING: dipakai `normalize-space(.)` (string-value SELURUH subtree
    # elemen), BUKAN `normalize-space(text())` (hanya teks anak LANGSUNG).
    # Streamlit membungkus label tombol di dalam elemen anak (mis.
    # <button><div><p>Label</p></div></button>), jadi `button/text()`
    # SELALU kosong -- predikat berbasis text() tidak akan pernah cocok
    # walau ditunggu berapa lama pun. Ini root cause mayoritas kegagalan.
    def click_button(self, text: str, timeout: int = None) -> None:
        xpath = f'//div[@data-testid="stButton"]//button[normalize-space(.)="{text}"]'
        btn = self.wait(timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
        btn.click()

    def button_exists(self, text: str, timeout: int = 3) -> bool:
        xpath = f'//div[@data-testid="stButton"]//button[normalize-space(.)="{text}"]'
        try:
            self.wait(timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
            return True
        except TimeoutException:
            return False

    def download_button_exists(self, text: str, timeout: int = 3) -> bool:
        xpath = f'//div[@data-testid="stDownloadButton"]//button[normalize-space(.)="{text}"]'
        try:
            self.wait(timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
            return True
        except TimeoutException:
            return False

    # ── Alert (st.error / st.warning / st.success) ─────────────────────

    def get_alert_texts(self, timeout: int = 3) -> List[str]:
        try:
            elems = self.wait(timeout).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '[data-testid="stAlert"]'))
            )
            return [e.text for e in elems]
        except TimeoutException:
            return []

    # ── Gambar (st.image / st.pyplot) ───────────────────────────────────

    def get_all_image_elements(self):
        return self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="stImage"] img')

    def get_image_pil(self, img_element, timeout: int = None) -> Optional[Image.Image]:
        self.wait(timeout).until(lambda d: img_element.get_attribute("src"))
        src = img_element.get_attribute("src")
        if src and src.startswith("data:image"):
            b64_data = src.split(",", 1)[1]
            raw = base64.b64decode(b64_data)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        png_bytes = img_element.screenshot_as_png
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")

    # ── Tabel metrik (st.dataframe / st.table) ─────────────────────────

    def get_dataframe_text(self) -> str:
        elems = self.driver.find_elements(
            By.CSS_SELECTOR, '[data-testid="stDataFrame"], [data-testid="stTable"]'
        )
        return "\n".join(e.text for e in elems)

    # ── Expander (dipakai app.py untuk membungkus tiap hasil per-citra) ──

    def get_expander_texts(self) -> List[str]:
        elems = self.driver.find_elements(By.CSS_SELECTOR, '[data-testid="stExpander"]')
        return [e.text for e in elems]

    # ── Fallback generik: seluruh teks halaman ──────────────────────────

    def get_page_text(self) -> str:
        app = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="stApp"]')
        return app.text

    def take_screenshot(self, name: str) -> str:
        """
        Screenshot HALAMAN PENUH, bukan cuma viewport yang sedang terlihat.
        driver.save_screenshot() bawaan hanya menangkap area yang sedang
        di-scroll ke tampilan -- kalau elemen yang relevan (mis. tombol
        'Jalankan Inferensi' di bawah form upload) ada di luar viewport
        saat itu, screenshot biasa tidak akan menunjukkannya. Di sini
        window di-resize dulu ke tinggi penuh konten sebelum mengambil
        gambar, lalu dikembalikan ke ukuran semula.
        """
        os.makedirs(settings.SCREENSHOT_DIR, exist_ok=True)
        path = os.path.join(settings.SCREENSHOT_DIR, f"{name}.png")
        try:
            width = self.driver.execute_script("return document.body.scrollWidth") or settings.WINDOW_SIZE[0]
            height = self.driver.execute_script("return document.body.scrollHeight") or settings.WINDOW_SIZE[1]
            original_size = self.driver.get_window_size()
            self.driver.set_window_size(width, height + 100)
            self.driver.save_screenshot(path)
            self.driver.set_window_size(original_size["width"], original_size["height"])
        except Exception:
            self.driver.save_screenshot(path)
        return path
