"""src/visualization.py — Visualisasi overlay TP/FP/FN (v3)."""
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

import config
from src.mask_utils import compute_tp_fp_fn


# ── Overlay ───────────────────────────────────────────────────────────────────

def overlay_with_label(image: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                       alpha: float = config.OVERLAY_ALPHA) -> np.ndarray:
    """TP=Hijau, FP=Kuning, FN=Merah di atas citra asli."""
    tp_map, fp_map, fn_map = compute_tp_fp_fn(pred, gt)
    out = image.astype(np.float32).copy()

    def _blend(region, color):
        for c, v in enumerate(color):
            out[:,:,c] = np.where(region,
                out[:,:,c]*(1-alpha) + v*alpha, out[:,:,c])

    _blend(fn_map, config.COLOR_FN)
    _blend(fp_map, config.COLOR_FP)
    _blend(tp_map, config.COLOR_TP)
    return out.clip(0,255).astype(np.uint8)


def overlay_prediction_only(image: np.ndarray, pred: np.ndarray,
                             alpha: float = config.OVERLAY_ALPHA) -> np.ndarray:
    """Overlay biru untuk prediksi tanpa label."""
    out = image.astype(np.float32).copy()
    region = pred.astype(bool)
    for c, v in enumerate(config.COLOR_PRED):
        out[:,:,c] = np.where(region, out[:,:,c]*(1-alpha)+v*alpha, out[:,:,c])
    return out.clip(0,255).astype(np.uint8)


# ── Figure ─────────────────────────────────────────────────────────────────────

def figure_with_legend(image: np.ndarray, pred: np.ndarray,
                       gt: Optional[np.ndarray], title: str = "",
                       alpha: float = config.OVERLAY_ALPHA) -> plt.Figure:
    has_gt = gt is not None
    n = 4 if has_gt else 3
    fig, axes = plt.subplots(1, n, figsize=(4.5*n, 4.5))
    fig.patch.set_facecolor("#1a1a1a")
    if title: fig.suptitle(title, fontsize=11, fontweight="bold", color="white")

    kw = {"color":"white","fontsize":9}

    axes[0].imshow(image); axes[0].set_title("Citra Asli", **kw); axes[0].axis("off")

    idx = 1
    if has_gt:
        axes[idx].imshow(gt*255, cmap="gray", vmin=0, vmax=255)
        axes[idx].set_title("Ground Truth", **kw); axes[idx].axis("off"); idx += 1

    axes[idx].imshow(pred*255, cmap="gray", vmin=0, vmax=255)
    axes[idx].set_title("Prediksi", **kw); axes[idx].axis("off"); idx += 1

    if has_gt:
        ov = overlay_with_label(image, pred, gt, alpha)
        axes[idx].imshow(ov); axes[idx].set_title("Overlay", **kw); axes[idx].axis("off")
        patches = [
            mpatches.Patch(facecolor=tuple(v/255 for v in config.COLOR_TP),
                           label="TP (Benar Positif)", linewidth=0),
            mpatches.Patch(facecolor=tuple(v/255 for v in config.COLOR_FP),
                           label="FP (Salah Positif)", linewidth=0),
            mpatches.Patch(facecolor=tuple(v/255 for v in config.COLOR_FN),
                           label="FN (Salah Negatif)", linewidth=0),
        ]
    else:
        ov = overlay_prediction_only(image, pred, alpha)
        axes[idx].imshow(ov); axes[idx].set_title("Overlay", **kw); axes[idx].axis("off")
        patches = [mpatches.Patch(facecolor=tuple(v/255 for v in config.COLOR_PRED),
                                  label="Prediksi (Sel)", linewidth=0)]

    for ax in axes: ax.set_facecolor("#1a1a1a")
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=8, framealpha=0.3, facecolor="#2a2a2a", labelcolor="white")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


def figure_to_bytes(fig: plt.Figure, dpi: int = 120) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ── Bar chart perbandingan ────────────────────────────────────────────────────

def comparison_bar_chart(records: list, metrics: list,
                         model_names: list) -> plt.Figure:
    import pandas as pd
    df = pd.DataFrame(records)
    n  = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    colors = {"SAM2": "#4C72B0", "U-Net": "#DD8452"}

    for ax, metric in zip(axes, metrics):
        means = {}
        for m in model_names:
            sub = df[df["model"] == m][metric].dropna()
            means[m] = sub.mean() if len(sub) else 0.0
        bars = ax.bar(list(means), list(means.values()),
                      color=[colors.get(m,"#888") for m in means],
                      edgecolor="black", linewidth=0.8)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(metric.replace("_"," ").title(), fontweight="bold")
        ax.set_ylim(0, max(1.05, max(means.values())*1.15) if means else 1)
        ax.set_ylabel("Nilai"); ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Perbandingan Metrik Antar Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig
