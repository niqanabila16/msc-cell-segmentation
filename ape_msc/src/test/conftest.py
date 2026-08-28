"""
conftest.py — Fixture bersama untuk seluruh test suite (unit + e2e).

tests/unit/  -> murni Python, TIDAK butuh app.py berjalan, TIDAK butuh
                browser. Menguji src/metrics.py & src/data_utils.py
                langsung sebagai modul (lihat "Generating application
                state" & Test Pyramid di README untuk alasannya).
tests/e2e/   -> butuh Streamlit AKTIF di testconfig.settings.BASE_URL:

    streamlit run app.py

lalu, dari root proyek (ape_msc/):

    pytest src/test                 # semua (unit + e2e)
    pytest src/test/tests/unit      # cuma unit, cepat, tanpa browser
    pytest src/test/tests/e2e       # cuma e2e
"""
import sys
from pathlib import Path

# src/test ada di sys.path -> `from testconfig import settings` dst selalu
# resolve terlepas dari direktori mana pytest dijalankan.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Root proyek (ape_msc/) JUGA ditambahkan -> supaya tests/unit/ bisa
# `import src.metrics`, `import src.data_utils` langsung tanpa Streamlit.
# Ini aman terhadap tabrakan nama: paket config kita sendiri sengaja
# bernama "testconfig" (bukan "config"), jadi `import config` di dalam
# src/data_utils.py tetap benar resolve ke ape_msc/config.py milik proyek.
PROJECT_ROOT = ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(1, str(PROJECT_ROOT))

import pytest  # noqa: E402
from selenium import webdriver  # noqa: E402
from selenium.webdriver.chrome.options import Options as ChromeOptions  # noqa: E402
from selenium.webdriver.firefox.options import Options as FirefoxOptions  # noqa: E402

from testconfig import settings  # noqa: E402
from pages.app_page import AppPage  # noqa: E402
from utils.test_data_generator import ensure_test_data  # noqa: E402


def _build_chrome_driver():
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = ChromeOptions()
    if settings.HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--window-size={settings.WINDOW_SIZE[0]},{settings.WINDOW_SIZE[1]}")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def _build_firefox_driver():
    from selenium.webdriver.firefox.service import Service
    from webdriver_manager.firefox import GeckoDriverManager

    opts = FirefoxOptions()
    if settings.HEADLESS:
        opts.add_argument("-headless")
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=service, options=opts)
    driver.set_window_size(*settings.WINDOW_SIZE)
    return driver


@pytest.fixture(scope="session")
def test_data() -> dict:
    """Bangkitkan semua file uji sintetis sekali per sesi test."""
    return ensure_test_data()


@pytest.fixture()
def driver():
    """WebDriver baru per test — memastikan tiap test terisolasi (state
    upload/slider/dropdown Streamlit tidak bocor ke test lain)."""
    if settings.BROWSER == "firefox":
        drv = _build_firefox_driver()
    else:
        drv = _build_chrome_driver()
    drv.implicitly_wait(0)  # kita pakai explicit wait di semua tempat
    yield drv
    drv.quit()


@pytest.fixture()
def app_page(driver) -> AppPage:
    page = AppPage(driver)
    page.open()
    return page


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Ambil screenshot HALAMAN PENUH otomatis kalau ada test yang gagal
    (lihat BasePage.take_screenshot() untuk kenapa halaman penuh, bukan
    viewport saja)."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        drv = item.funcargs.get("driver")
        if drv is not None:
            from pages.base_page import BasePage

            try:
                path = BasePage(drv).take_screenshot(item.name.replace("/", "_").replace("::", "__"))
                print(f"\n[screenshot halaman penuh] Kegagalan disimpan di: {path}")
            except Exception:
                pass
