"""
src/ui_helpers.py — Komponen UI Streamlit (v10).

Perubahan dari v9:
- sidebar_poly_controls(): dropdown stage sekarang 2 opsi saja.
- show_poly_intermediate_stages(): diperbarui ke 2 stage.
- Hapus referensi normalisasi koordinat sebagai stage terpisah.
"""

from io import BytesIO
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import config

from src.image_utils import (
    PREPROCESSING_OPTIONS,
    POLY_STAGE_LABELS,
    POLY_STAGE_DESC,
    POLY_STAGE_ROLES,
    get_poly_intermediate_stages,
    rgb_to_gray,
)


# ─────────────────────────────────────────────────────────────────────────────
# Format helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(val, d=4, sfx="") -> str:
    return f"{val:.{d}f}{sfx}" if val is not None else "—"


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame builders
# ─────────────────────────────────────────────────────────────────────────────

def metrics_to_df(metrics: dict) -> pd.DataFrame:
    rows = [
        ("IoU",                       fmt(metrics.get("iou"))),
        ("Dice / F1",                 fmt(metrics.get("dice"))),
        ("Precision",                 fmt(metrics.get("precision"))),
        ("Recall",                    fmt(metrics.get("recall"))),
        ("Konfluensi Prediksi (%)",   fmt(metrics.get("confluence_pred"),   2, "%")),
        ("Konfluensi Aktual (%)",     fmt(metrics.get("confluence_actual"), 2, "%")),
        ("Waktu Inferensi",           fmt(metrics.get("inference_time_sec"), 4, " dtk")),
    ]
    return pd.DataFrame(rows, columns=["Metrik", "Nilai"])


def avg_metrics_df(mlist: list) -> pd.DataFrame:
    from src.metrics import average_metrics
    avg = average_metrics(mlist)
    rows = [
        ("Rata-rata IoU",              fmt(avg.get("iou"))),
        ("Rata-rata Dice Coefficient", fmt(avg.get("dice_coefficient"))),
        ("Rata-rata Waktu Inferensi",  fmt(avg.get("inference_time_sec"), 4, " dtk")),
    ]
    return pd.DataFrame(rows, columns=["Metrik", "Nilai"])


def records_to_summary_df(records: list) -> pd.DataFrame:
    rename = {
        "nama_citra":        "Nama Citra",
        "model":             "Model",
        "iou":               "IoU",
        "dice":              "Dice",
        "precision":         "Precision",
        "recall":            "Recall",
        "confluence_pred":   "Konfluensi Pred (%)",
        "confluence_actual": "Konfluensi Aktual (%)",
        "percentage_error":  "% Error",
        "inference_time_sec":"Waktu (dtk)",
    }
    df   = pd.DataFrame(records)
    drop = [c for c in ("warnings", "vis_bytes", "run_id", "path_output") if c in df]
    df   = df.drop(columns=drop).rename(
        columns={k: v for k, v in rename.items() if k in df}
    )
    for col in ["IoU", "Dice", "Precision", "Recall"]:
        if col in df:
            df[col] = df[col].apply(lambda x: round(x, 4) if x else None)
    for col in ["Konfluensi Pred (%)", "Konfluensi Aktual (%)", "% Error"]:
        if col in df:
            df[col] = df[col].apply(lambda x: round(x, 2) if x else None)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Notifikasi
# ─────────────────────────────────────────────────────────────────────────────

def show_error(msg: str):   st.error(f" {msg}")
def show_warning(msg: str): st.warning(f" {msg}")
def show_success(msg: str): st.success(f" {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Download buttons
# ─────────────────────────────────────────────────────────────────────────────

def download_png(img_bytes: bytes, filename: str,
                 label="Unduh Visualisasi", key=None):
    st.download_button(label, img_bytes, filename, "image/png", key=key)


def download_csv(df: pd.DataFrame, filename: str,
                 label="Unduh CSV", key=None):
    st.download_button(
        label, df.to_csv(index=False).encode("utf-8"),
        filename, "text/csv", key=key,
    )


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    mask = (mask > 0).astype("uint8") * 255
    img  = Image.fromarray(mask, mode="L")
    img  = img.resize(
        (img.width * 2, img.height * 2),
        resample=Image.Resampling.LANCZOS,
    )
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def download_mask_png(mask: np.ndarray, filename: str,
                      label="Unduh Mask", key=None):
    st.download_button(
        label, mask_to_png_bytes(mask), filename, "image/png", key=key,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar widgets
# ─────────────────────────────────────────────────────────────────────────────

def sidebar_alpha() -> float:
    return st.sidebar.slider(
        "Transparansi Overlay", 0.1, 0.9,
        config.OVERLAY_ALPHA, 0.05, key="alpha",
    )


def sidebar_preprocessing() -> str:
    label = st.sidebar.selectbox(
        "preprocessing Citra",
        list(PREPROCESSING_OPTIONS.keys()),
        key="prep",
    )
    return PREPROCESSING_OPTIONS[label]


def sidebar_poly_controls() -> None:
    """
    Dropdown tahap polynomial (1–2), selalu tampil di bawah preprocessing.
    Orde dikunci ke 3 (Li et al., 2025 — MAE terendah).
    Normalisasi koordinat diterapkan internal pada kedua stage.
    Update config.POLY_STAGE secara langsung.
    """
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("**Konfigurasi Polynomial** *(orde 3)*")

    stage_labels = list(POLY_STAGE_LABELS.values())
    stage_keys   = list(POLY_STAGE_LABELS.keys())

    current_stage = int(getattr(config, "POLY_STAGE", 2))
    current_idx   = (
        stage_keys.index(current_stage)
        if current_stage in stage_keys else 1
    )

    # selected_label = st.sidebar.selectbox(
    #     "Tahap Polynomial",
    #     stage_labels,
    #     index=current_idx,
    #     key="poly_stage",
    #     help=(
    #         "Stage 1: dasar — fit + divide per Li et al. (2025).\n"
    #         "Stage 2: full — tambahan normalize ke [0,1] "
    #         "(direkomendasikan jika dikombinasikan dengan Percentile)."
    #     ),
    # )
    # selected_stage = stage_keys[stage_labels.index(selected_label)]
    # config.POLY_STAGE = selected_stage

    # st.sidebar.caption(POLY_STAGE_DESC.get(selected_stage, ""))


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial intermediate stages viewer
# ─────────────────────────────────────────────────────────────────────────────

def show_poly_intermediate_stages(
    image: np.ndarray,
    expander_label: str = "Tahapan Polynomial Background Correction",
) -> None:
    """
    Tampilkan citra intermediate setiap tahap polynomial dalam st.expander.
    Untuk dokumentasi laporan: membuktikan efek setiap langkah.

    Parameters
    ----------
    image : np.ndarray — uint8 RGB (H, W, 3) atau grayscale (H, W)
    """
    if image.ndim == 3:
        gray_01 = rgb_to_gray(image)
    else:
        gray_01 = image.astype(np.float32) / 255.0

    stage_max = int(getattr(config, "POLY_STAGE", 2))
    stages    = get_poly_intermediate_stages(gray_01, stage_max=stage_max)

    with st.expander(expander_label, expanded=False):
        st.caption(
            f"Orde: **3** (dikunci) | "
            f"Tahap aktif: **{stage_max} — {POLY_STAGE_LABELS.get(stage_max, '')}**"
        )

        # Grid: input + stage(s) dalam satu baris
        items = list(stages.items())
        cols  = st.columns(len(items))
        for col, (label, img_arr) in zip(cols, items):
            img_u8 = np.clip(img_arr * 255.0, 0, 255).round().astype(np.uint8)
            with col:
                st.image(img_u8, caption=label, use_container_width=True)

        st.markdown("---")

        # Tabel statistik distribusi per tahap
        st.markdown("**Statistik distribusi per tahap:**")
        stat_rows = []
        for label, img_arr in stages.items():
            stat_rows.append({
                "Tahap": label,
                "Min":   f"{img_arr.min():.4f}",
                "Max":   f"{img_arr.max():.4f}",
                "Mean":  f"{img_arr.mean():.4f}",
                "Std":   f"{img_arr.std():.4f}",
            })
        st.dataframe(
            pd.DataFrame(stat_rows),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("---")

        # Peran dan deskripsi tiap stage yang aktif
        st.markdown("**Peran dan justifikasi setiap tahap:**")
        for stage_num, lbl in POLY_STAGE_LABELS.items():
            if stage_num > stage_max:
                continue
            role = POLY_STAGE_ROLES.get(stage_num, "")
            desc = POLY_STAGE_DESC.get(stage_num, "")
            st.markdown(f"**{lbl}**")
            st.caption(f"*Peran:* {role}")
            st.caption(f"*Implementasi:* {desc}")

        # Tabel MAE per orde untuk laporan
        st.markdown("---")
        st.markdown("**Justifikasi orde 3 (Li et al., 2025):**")
        st.markdown("""
| Orde | Basis Term | MAE |
|------|-----------|-----|
| 1 | 3 | 0.927 |
| 2 | 6 | 0.495 |
| **3** | **10** | **0.482** |
| 4 | 15 | 0.497 |
        """)