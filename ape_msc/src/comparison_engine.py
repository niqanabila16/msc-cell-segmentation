"""src/comparison_engine.py — Perbandingan dua model (v3, tanpa history)."""
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from src.inference_engine import run_inference
from src.metrics import compute_all_metrics
from src.visualization import figure_to_bytes, figure_with_legend


def run_comparison(
    model_a_name: str, model_a_inst,
    model_b_name: str, model_b_inst,
    images: list, image_names: list,
    labels: Optional[list] = None,
    preprocessing: str = "none",
    alpha: float = config.OVERLAY_ALPHA,
) -> dict:
    run_id = str(uuid.uuid4())
    labels  = labels or [None] * len(images)
    records = []
    vis_bytes: dict = {}

    for img, name, gt in zip(images, image_names, labels):
        for m_name, m_inst in [(model_a_name, model_a_inst),
                                (model_b_name, model_b_inst)]:
            res     = run_inference(m_name, m_inst, img, preprocessing)
            mask    = res["mask"]
            t       = res["time_sec"]
            metrics = compute_all_metrics(mask, gt, t)

            fig = figure_with_legend(img, mask, gt,
                                     title=f"{name} — {m_name}", alpha=alpha)
            vb  = figure_to_bytes(fig)

            out_path = config.COMPARISON_DIR / f"{Path(name).stem}_{m_name}_{run_id[:6]}.png"
            out_path.write_bytes(vb)

            vis_bytes[f"{name}|{m_name}"] = vb
            records.append({
                "run_id": run_id, "nama_citra": name, "model": m_name,
                **metrics, "path_output": str(out_path),
                "warnings": res["warnings"], "vis_bytes": vb,
            })

    df = pd.DataFrame(records)
    numeric = ["iou","dice","precision","recall","confluence_pred",
               "confluence_actual","percentage_error","inference_time_sec"]
    summary = df.groupby("model")[[c for c in numeric if c in df]].mean()

    return {"records": records, "df": df, "summary": summary,
            "run_id": run_id, "vis_bytes": vis_bytes}
