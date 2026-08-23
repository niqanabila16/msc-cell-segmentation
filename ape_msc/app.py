"""
app.py — APE MSC.
Jalankan: streamlit run app.py

Perubahan v7:
- SAM2 dihapus sepenuhnya: dropdown "Jenis Model" (dulu SAM2/U-Net),
  konversi grayscale->RGB khusus SAM2, caption "SAM2 input: 1024x1024".
  Aplikasi sekarang hanya menjalankan U-Net (vanilla atau SMP ResNet50,
  keduanya sudah otomatis terdeteksi dari checkpoint di model_loader.py).
- resize_for_unet (pad+resize kotak) dihapus dari inference_engine.py —
  itu kebutuhan SAM2. U-Net di sini fully-convolutional, jadi tinggal
  diskalakan proporsional sesuai model._input_scale.
"""

import traceback
from pathlib import Path
from typing import Optional
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import uuid

import config
from src.data_utils import (
    validate_image_file, validate_label_file,
    load_image_from_bytes, load_label_from_bytes,
)
from src.image_utils import ensure_rgb, apply_preprocessing
from src.inference_engine import run_inference
from src.metrics import compute_all_metrics
from src.model_loader import load_unet
from src.visualization import figure_with_legend, figure_to_bytes

from src.ui_helpers import (
    metrics_to_df, avg_metrics_df, records_to_summary_df,
    show_error, show_warning, show_success,
    download_png, download_csv, download_mask_png,
    sidebar_alpha, sidebar_preprocessing, sidebar_poly_controls,
    show_poly_intermediate_stages,
    PREPROCESSING_OPTIONS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title=config.APP_TITLE, page_icon=config.APP_ICON,
                   layout="wide")

_PREPROCESS_LABEL: dict = {v: k for k, v in PREPROCESSING_OPTIONS.items()}

_SLOW_METHODS = {"poly", "poly_percentile"}

MODEL_NAME = "U-Net"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_model(ckpt_path: str):
    return load_unet(ckpt_path)


def _get_model(ver: dict):
    ckpt = ver["path"]
    try:
        return _load_model(ckpt)
    except FileNotFoundError as e:
        raise RuntimeError(str(e))
    except ImportError as e:
        raise RuntimeError(
            f"Gagal mengimport dependency untuk U-Net:\n{e}\n\n"
            f"Kemungkinan package yang hilang (coba pip install):\n"
            f"  six\n  segmentation-models-pytorch\n\n"
            f"Detail:\n{traceback.format_exc()}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Gagal memuat model:\n{e}\n\n"
            f"Detail:\n{traceback.format_exc()}"
        )


def _resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
    )


def _parse_uploads(imgs_up, lbls_up) -> tuple[list, list, list, list]:
    images, names, gts, errs = [], [], [], []

    master_json = None
    if lbls_up:
        for lbl_file in lbls_up:
            if lbl_file.name.lower().endswith(".json"):
                master_json = lbl_file
                break

    for img_file in imgs_up:
        name = img_file.name
        try:
            img_arr = load_image_from_bytes(img_file.getvalue())
        except Exception as e:
            errs.append(f"Gagal memuat citra '{name}': {e}")
            continue

        images.append(img_arr)
        names.append(name)

        gt = None
        if lbls_up:
            target_stem = Path(name).stem
            paired_png = next(
                (l for l in lbls_up
                 if (Path(l.name).stem == target_stem
                     or Path(l.name).stem == f"{target_stem}_mask")
                 and l.name.lower().endswith(".png")),
                None,
            )

            if paired_png:
                try:
                    gt = load_label_from_bytes(
                        file_bytes=paired_png.getvalue(),
                        filename=paired_png.name,
                        image_shape=img_arr.shape[:2],
                    )
                except Exception as e:
                    errs.append(
                        f"[{name}] Gagal memuat mask PNG '{paired_png.name}': {e}"
                    )
            elif master_json:
                try:
                    gt = load_label_from_bytes(
                        file_bytes=master_json.getvalue(),
                        filename=master_json.name,
                        image_shape=img_arr.shape[:2],
                        target_image_name=name,
                    )
                    if gt is not None and gt.max() == 0:
                        errs.append(
                            f"[{name}] Tidak ditemukan dalam annotations.json → GT diabaikan."
                        )
                        gt = None
                except Exception as e:
                    errs.append(f"[{name}] Gagal ekstrak dari annotations.json: {e}")

        gts.append(gt)

    return images, names, gts, errs


# ── Preview preprocessing ───────────────────────────────────────────────────────

_MAX_PREVIEW = 8


def _render_preprocessing_preview(uploaded_files: list, method: str) -> None:
    method_label = _PREPROCESS_LABEL.get(method, method)

    if method == "none":
        st.info(
            "Preprocessing aktif: **Tanpa preprocessing** — "
            "citra dikirim langsung ke model tanpa perubahan."
        )
        return

    if method in _SLOW_METHODS:
        st.caption(
            "⏳ Metode ini menggunakan fitting polinomial — "
            "preview mungkin membutuhkan beberapa detik per citra."
        )

    files_to_show = uploaded_files[:_MAX_PREVIEW]
    n_remaining   = len(uploaded_files) - len(files_to_show)

    if n_remaining > 0:
        st.caption(
            f"Menampilkan **{len(files_to_show)}** dari "
            f"**{len(uploaded_files)}** citra. "
            f"{n_remaining} citra lainnya diproses saat inferensi."
        )

    previews = []
    with st.spinner(f"Menerapkan preprocessing '{method_label}'…"):
        for f in files_to_show:
            try:
                img     = load_image_from_bytes(f.getvalue())
                img_rgb = ensure_rgb(img)
                proc    = apply_preprocessing(img_rgb, method)
                previews.append((f.name, img_rgb, proc))
            except Exception as exc:
                st.warning(f"Preview '{f.name}' gagal: {exc}")

    if not previews:
        st.warning("Tidak ada citra yang berhasil diproses untuk preview.")
        return

    for i in range(0, len(previews), 2):
        outer_l, outer_r = st.columns(2, gap="medium")
        for j, outer_col in enumerate((outer_l, outer_r)):
            if i + j >= len(previews):
                break
            name, original, preprocessed = previews[i + j]
            with outer_col:
                with st.container(border=True):
                    st.caption(f"📄 **{name}**")
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.image(original, caption="Asli",
                                 use_container_width=True)
                    with col_after:
                        st.image(preprocessed, caption=method_label,
                                 use_container_width=True)

    if "poly" in method and previews:
        # Gunakan citra pertama sebagai contoh untuk intermediate stages
        _, first_img_rgb, _ = previews[0]
        show_poly_intermediate_stages(
            first_img_rgb,
            expander_label=f"Tahapan Polynomial — {previews[0][0]}",
        )


def _preview_section(uploaded_files, method: str, toggle_key: str) -> None:
    if not uploaded_files:
        return

    st.divider()

    col_tog, col_info = st.columns([2, 5])
    with col_tog:
        show = st.toggle(
            "Preview Preprocessing",
            value=False,
            key=toggle_key,
            help="Tampilkan perbandingan citra asli vs citra setelah preprocessing.",
        )
    with col_info:
        if show:
            st.caption(
                f"Metode aktif: **{_PREPROCESS_LABEL.get(method, method)}** |  "
                f"{len(uploaded_files)} citra diupload"
            )

    if show:
        _render_preprocessing_preview(uploaded_files, method)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(f"{config.APP_ICON} APE MSC")
    st.caption("Aplikasi Pendukung Eksperimen\nSemantic Segmentation MSC")
    st.divider()

    st.subheader("Preprocessing")
    preprocessing = sidebar_preprocessing()

    sidebar_poly_controls()

    st.divider()
    st.subheader("Visualisasi")
    alpha = sidebar_alpha()
    config.OVERLAY_ALPHA = alpha

    st.divider()
    st.caption(
        "Batas upload: **10 MB** / file  \n"
        "Format citra: **.jpg / .jpeg / .png** \n"
        "Label: **.png / .json** (nama file sama)"
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.title(f"{config.APP_ICON} {config.APP_TITLE}")
st.caption("Evaluasi U-Net pada citra mikroskopis MSC.")
st.divider()

tab_infer = st.tabs(["Inferensi"])[0]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INFERENSI
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    st.header("Inferensi Model")

    vers_i = config.UNET_VERSIONS
    labs_i = [v["label"] for v in vers_i]
    sel_i  = st.selectbox("Versi / Konfigurasi", labs_i, key="i_ver")
    vd_i   = vers_i[labs_i.index(sel_i)]

    st.info(
        f"**Model:** {MODEL_NAME}  |  **Versi:** {sel_i}  |  "
        f"**Preprocessing:** {_PREPROCESS_LABEL.get(preprocessing, preprocessing)}  |  "
        f"**Path:** `{vd_i['path']}`"
    )
    st.divider()

    st.subheader("Upload Citra")
    imgs_up = st.file_uploader(
        "Pilih satu atau lebih file citra",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="i_imgs",
    )

    st.subheader("Upload Label GT (opsional)")
    st.caption(
        "Nama file label harus sama dengan citra. "
        "Contoh: `sel_001.jpg` → `sel_001_mask.png` atau `annotations.json` "
        "dengan entry untuk `sel_001.jpg`."
    )
    lbls_up = st.file_uploader(
        "Pilih file label (.png / .json)",
        type=["png", "json"],
        accept_multiple_files=True,
        key="i_lbls",
    )

    _preview_section(imgs_up, preprocessing, toggle_key="i_prev")

    st.divider()
    run_btn = st.button("Jalankan Inferensi", type="primary",
                        use_container_width=True)
    avg_ph = st.empty()

    if run_btn:
        if not imgs_up:
            show_error("Belum ada citra yang di-upload."); st.stop()

        with st.spinner(f"Memuat {MODEL_NAME} — {sel_i}…"):
            try:
                model_inst = _get_model(vd_i)
            except RuntimeError as e:
                show_error(str(e)); st.stop()

        images, names, gts, errs = _parse_uploads(imgs_up, lbls_up)
        for e in errs: show_warning(e)
        if not images:
            show_error("Tidak ada citra valid."); st.stop()

        prog = st.progress(0, text="Memulai inferensi…")
        all_results, all_metrics = [], []

        for idx, (img, name, gt) in enumerate(zip(images, names, gts)):
            prog.progress(idx / len(images),
                          text=f"({idx+1}/{len(images)}) {name}…")

            try:
                res = run_inference(model_inst, img, preprocessing)
            except Exception as e:
                show_error(f"[{name}] Inferensi gagal: {e}")
                with st.expander("Detail error"):
                    st.code(traceback.format_exc())
                continue

            mask_eval    = res["mask"]
            mask_display = res.get("mask_display", mask_eval)
            t            = res["time_sec"]

            gt_eval = None
            if gt is not None:
                gt_eval = gt
                if gt_eval.shape != mask_eval.shape:
                    gt_eval = _resize_mask_nearest(gt_eval, mask_eval.shape)

            metrics = compute_all_metrics(mask_eval, gt_eval, t)
            all_metrics.append(metrics)

            fig = figure_with_legend(
                img, mask_display, gt,
                title=f"{name} — {MODEL_NAME}",
                alpha=alpha,
            )
            vb       = figure_to_bytes(fig)
            out_path = config.INFERENCE_DIR / f"{Path(name).stem}_{MODEL_NAME}.png"
            out_path.write_bytes(vb)

            all_results.append({
                "name": name, "metrics": metrics,
                "vis_bytes": vb, "gt": gt,
                "mask_eval": mask_eval, "mask_display": mask_display,
            })

        prog.progress(1.0, text="Selesai!"); prog.empty()

        if not all_results:
            show_error("Tidak ada hasil yang berhasil diproses."); st.stop()

        show_success(f"{len(all_results)} citra berhasil diproses.")

        if any(m.get("iou") is not None for m in all_metrics):
            with avg_ph.container():
                st.subheader("Rata-rata Hasil Batch")
                st.dataframe(avg_metrics_df(all_metrics),
                             use_container_width=True, hide_index=True)
        else:
            avg_ph.info(
                "Label tidak ditemukan → rata-rata IoU/Dice/Error tidak tersedia."
            )

        st.divider()

        for rec in all_results:
            with st.expander(f"{rec['name']}", expanded=True):
                c_vis, c_met = st.columns([2, 1])
                with c_vis:
                    st.image(rec["vis_bytes"],
                             caption="Citra Asli | GT | Prediksi | Overlay",
                             use_container_width=True)
                    unique_btn_id = str(uuid.uuid4())
                    download_png(
                        rec["vis_bytes"],
                        f"{Path(rec['name']).stem}_{MODEL_NAME}.png",
                        key=f"dl_png_{unique_btn_id}",
                    )
                    download_mask_png(
                        rec["mask_display"],
                        f"{Path(rec['name']).stem}_{MODEL_NAME}_mask.png",
                        label="Unduh Mask Hitam-Putih",
                        key=f"dl_mask_{unique_btn_id}",
                    )
                with c_met:
                    st.markdown("**Metrik**")
                    st.dataframe(metrics_to_df(rec["metrics"]),
                                 use_container_width=True, hide_index=True)

        if len(all_results) > 1:
            st.divider()
            clean = [
                {"nama_citra": r["name"], "model": MODEL_NAME, **r["metrics"]}
                for r in all_results
            ]
            download_csv(
                records_to_summary_df(clean),
                "hasil_inferensi.csv",
                "Unduh Semua Hasil (CSV)",
            )
