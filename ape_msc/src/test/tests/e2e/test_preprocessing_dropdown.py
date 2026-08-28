"""
tests/e2e/test_preprocessing_dropdown.py — FR10.

Sama seperti test_overlay_alpha_slider.py: presence/opsi tidak butuh
inferensi (murah), hanya test terakhir yang perlu dua kali run untuk
membuktikan dropdown benar-benar mempengaruhi hasil.
"""
import pytest

from testconfig import settings


pytestmark = pytest.mark.fr10


def test_preprocessing_dropdown_present_with_all_options(app_page, subtests):
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC

    with subtests.test(msg="dropdown ditemukan"):
        app_page.find_widget_by_label(settings.SELECTBOX_PREPROCESSING_LABEL, "stSelectbox")

    with subtests.test(msg="semua opsi tersedia"):
        container = app_page.find_widget_by_label(
            settings.SELECTBOX_PREPROCESSING_LABEL, "stSelectbox"
        )
        select_box = container.find_element(By.CSS_SELECTOR, '[data-baseweb="select"]')
        select_box.click()
        listbox = app_page.wait().until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'ul[role="listbox"]'))
        )
        option_texts = [
            o.text.strip() for o in listbox.find_elements(By.CSS_SELECTOR, 'li, [role="option"]')
        ]
        missing = [opt for opt in settings.PREPROCESSING_OPTIONS if opt not in option_texts]
        assert not missing, f"Opsi hilang: {missing}. Tersedia: {option_texts}"
        select_box.send_keys(Keys.ESCAPE)


@pytest.mark.parametrize("option_text", settings.PREPROCESSING_OPTIONS)
def test_each_preprocessing_option_can_be_selected(app_page, option_text):
    app_page.select_dropdown_option(settings.SELECTBOX_PREPROCESSING_LABEL, option_text)
    current = app_page.get_selected_dropdown_value(settings.SELECTBOX_PREPROCESSING_LABEL)
    assert current == option_text


def test_preprocessing_preview_toggle_appears_after_upload(app_page, test_data):
    from selenium.webdriver.common.by import By

    app_page.upload_image(test_data["matched_image"])
    toggles = app_page.driver.find_elements(
        By.XPATH, '//label[.//*[normalize-space(text())="Preview Preprocessing"]]'
    )
    assert toggles


def test_changing_preprocessing_then_rerunning_changes_result(app_page, test_data):
    """Satu-satunya test FR10 yang menjalankan inferensi (dua kali)."""
    app_page.upload_image(test_data["matched_image"])
    app_page.upload_label(test_data["matched_label"])

    app_page.select_dropdown_option(settings.SELECTBOX_PREPROCESSING_LABEL, "Tanpa Preprocessing")
    app_page.run_inference()
    img_none = app_page.get_image_pil(app_page.get_result_figure_image())
    assert img_none is not None

    app_page.select_dropdown_option(
        settings.SELECTBOX_PREPROCESSING_LABEL, "Polynomial + Percentile"
    )
    app_page.run_inference()
    img_poly = app_page.get_image_pil(app_page.get_result_figure_image())
    assert img_poly is not None

    import numpy as np

    a = np.array(img_none.resize((150, 150)))
    b = np.array(img_poly.resize((150, 150)))
    mean_abs_diff = float(np.abs(a.astype(int) - b.astype(int)).mean())
    assert mean_abs_diff > 0.5, (
        "Mengganti preprocessing lalu menjalankan ulang inferensi tidak "
        "menghasilkan perbedaan visual pada hasil."
    )
