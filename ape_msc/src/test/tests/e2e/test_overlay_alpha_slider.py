"""
tests/e2e/test_overlay_alpha_slider.py — FR09.

Presence/range/set-value TIDAK butuh inferensi (murah, dipisah).
Hanya SATU test (di paling bawah) yang perlu inferensi dua kali, karena
itu satu-satunya cara membuktikan slider benar-benar mempengaruhi
visualisasi (figure cuma dihitung ulang saat tombol run diklik lagi --
lihat app.py, figure_with_legend() ada di dalam blok `if run_btn:`).
"""
import pytest

from testconfig import settings


pytestmark = pytest.mark.fr09


def test_alpha_slider_present_and_default_in_range(app_page, subtests):
    with subtests.test(msg="slider ditemukan"):
        app_page.find_widget_by_label(settings.SLIDER_ALPHA_LABEL, "stSlider")
    with subtests.test(msg="nilai default dalam rentang konfigurasi"):
        value = app_page.get_slider_value(settings.SLIDER_ALPHA_LABEL)
        assert settings.SLIDER_ALPHA_MIN <= value <= settings.SLIDER_ALPHA_MAX


@pytest.mark.parametrize("target_value", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_alpha_slider_can_be_set_to_target_value(app_page, target_value):
    result = app_page.set_slider_value(
        settings.SLIDER_ALPHA_LABEL,
        target_value,
        settings.SLIDER_ALPHA_MIN,
        settings.SLIDER_ALPHA_MAX,
        settings.SLIDER_ALPHA_STEP,
    )
    assert result == pytest.approx(target_value, abs=settings.SLIDER_ALPHA_STEP / 2)


def test_changing_alpha_then_rerunning_changes_overlay_visualization(app_page, test_data):
    """Satu-satunya test FR09 yang menjalankan inferensi (dua kali, dalam
    SATU browser session -- bukan dua test terpisah dengan dua browser)."""
    app_page.upload_image(test_data["matched_image"])
    app_page.upload_label(test_data["matched_label"])
    app_page.run_inference()
    panel_low = app_page.get_overlay_panel_image()
    assert panel_low is not None

    app_page.set_slider_value(
        settings.SLIDER_ALPHA_LABEL,
        settings.SLIDER_ALPHA_MAX,
        settings.SLIDER_ALPHA_MIN,
        settings.SLIDER_ALPHA_MAX,
        settings.SLIDER_ALPHA_STEP,
    )
    app_page.run_inference()
    panel_high = app_page.get_overlay_panel_image()
    assert panel_high is not None

    import numpy as np

    arr_low = np.array(panel_low.resize((100, 100)))
    arr_high = np.array(panel_high.resize((100, 100)))
    mean_abs_diff = float(np.abs(arr_low.astype(int) - arr_high.astype(int)).mean())
    assert mean_abs_diff > 0.5, (
        "Mengubah slider dari minimum ke maksimum lalu menjalankan ulang "
        "inferensi tidak menghasilkan perubahan visual pada overlay."
    )
