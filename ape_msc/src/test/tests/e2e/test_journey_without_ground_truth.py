"""
tests/e2e/test_journey_without_ground_truth.py — FR01, FR05, FR06, FR07, FR08
untuk jalur TANPA ground truth. Sama seperti test_journey_with_ground_truth.py,
satu upload+inferensi dibagi ke beberapa subtest.
"""
import pytest

from testconfig import settings


pytestmark = pytest.mark.fr05


def test_full_journey_without_ground_truth(app_page, test_data, subtests):
    app_page.upload_image(test_data["unmatched_image"])
    app_page.run_inference()

    with subtests.test(msg="FR01/FR05: citra tanpa label tetap diproses"):
        assert app_page.inference_ran_successfully()
        assert app_page.result_exists_for(f"{test_data['unmatched_stem']}.jpg")
        fatal = [a for a in app_page.get_alert_texts(timeout=2) if "traceback" in a.lower()]
        assert not fatal, f"Muncul error: {fatal}"

    with subtests.test(msg="FR06: rata-rata batch dilaporkan tidak tersedia"):
        assert app_page.avg_metrics_unavailable_shown(), (
            f"Tanpa label sama sekali, seharusnya muncul info yang memuat "
            f"'{settings.MSG_AVG_METRICS_UNAVAILABLE_SUBSTR}'."
        )

    with subtests.test(msg="FR06: baris metrik IoU tidak dipaksakan berupa angka"):
        text = app_page.get_metrics_text()
        if "IoU" in text:
            assert "—" in text or "None" not in text

    with subtests.test(msg="FR07: gambar hasil ter-render dengan 3 panel"):
        ratio = app_page.get_result_figure_aspect_ratio()
        n = settings.N_PANELS_WITHOUT_GT
        assert ratio is not None
        assert (n - 1.3) <= ratio <= (n + 0.7), f"Rasio {ratio:.2f} tidak cocok {n} panel."

    with subtests.test(msg="FR08: overlay prediksi-saja tetap berwarna (bukan grayscale)"):
        import numpy as np

        panel = app_page.get_overlay_panel_image(n_panels=settings.N_PANELS_WITHOUT_GT)
        assert panel is not None
        arr = np.array(panel)
        channel_diff = np.abs(arr[..., 0].astype(int) - arr[..., 2].astype(int)).mean()
        assert channel_diff > 0.5, (
            "Panel overlay tanpa label tampak grayscale murni, padahal "
            "seharusnya ada overlay warna prediksi (biru)."
        )
