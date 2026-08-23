"""
src/metrics.py — Metrik evaluasi segmentasi (v3).
Dioptimalkan dan diselaraskan secara harfiah dengan narasi rumus laporan.
"""

from typing import Optional
import numpy as np
import config

def _check_same_shape(pred: np.ndarray, gt: np.ndarray) -> None:
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape pred dan gt harus sama untuk evaluasi. "
            f"Dapat pred={pred.shape}, gt={gt.shape}"
        )

def precision_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Precision: rasio jumlah sampel piksel positif yang berhasil diprediksi 
    secara benar (TP) terhadap total keseluruhan tebakan positif (TP + FP).
    """
    p, g = pred.astype(bool), gt.astype(bool)
    tp = (p & g).sum()
    fp = (p & ~g).sum()
    
    return float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Recall (Sensitivitas): proporsi sampel positif yang diprediksi benar (TP)
    terhadap keseluruhan piksel positif secara aktual (TP + FN).
    """
    p, g = pred.astype(bool), gt.astype(bool)
    tp = (p & g).sum()
    fn = (~p & g).sum()
    
    return float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Dice Coefficient (F1-Score): Merujuk pada Rumus (II.1).
    Dihitung persis menggunakan pendekatan rata-rata harmonik (harmonic mean)
    dari precision dan recall.
    """
    # Memanggil nilai Precision dan Recall sebagai komponen pelengkap
    precision = precision_score(pred, gt)
    recall = recall_score(pred, gt)
    
    # 2 * (Precision * Recall) / (Precision + Recall)
    penjumlah = precision + recall
    
    return float(2 * precision * recall) / float(penjumlah) if penjumlah > 0 else 0.0

def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    IoU / Jaccard Index: Merujuk pada Rumus (II.2).
    Rasio luas area irisan (intersection) terhadap area gabungan (union).
    """
    p, g = pred.astype(bool), gt.astype(bool)
    inter = (p & g).sum()
    union = (p | g).sum()
    
    return float(inter) / float(union) if union > 0 else (1.0 if inter == 0 else 0.0)

def confluence_percent(mask: np.ndarray) -> float:
    """
    Confluence Estimation Percentage: Merujuk pada Rumus (II.3).
    Total piksel prediksi sel dibagi total piksel keseluruhan dikali 100%.
    """
    if mask.size == 0:
        return 0.0
    return float(np.sum(mask > 0)) / float(mask.size) * 100.0

def percentage_error(pred_conf: float, actual_conf: float) -> float:
    """
    Percentage Error (PE): Merujuk pada Rumus (II.6).
    Selisih mutlak area prediksi dengan ground truth, dibagi ground truth, dikali 100%.
    """
    denom = max(actual_conf, config.EPSILON)
    return abs(pred_conf - actual_conf) / denom * 100.0

def compute_all_metrics(pred: np.ndarray, gt: Optional[np.ndarray], t: float) -> dict:
    """Agregator evaluasi metrik."""
    conf_pred = confluence_percent(pred)
    
    if gt is not None:
        _check_same_shape(pred, gt)
        iou      = iou_score(pred, gt)
        prec     = precision_score(pred, gt)
        rec      = recall_score(pred, gt)
        dice     = dice_score(pred, gt) # Menghitung Dice setelah Prec & Rec selesai dieksekusi
        conf_act = confluence_percent(gt)
        pct_err  = percentage_error(conf_pred, conf_act)
    else:
        iou = dice = prec = rec = conf_act = pct_err = None
        
    return {
        "iou": iou, 
        "dice": dice, 
        "precision": prec, 
        "recall": rec,
        "confluence_pred": conf_pred, 
        "confluence_actual": conf_act,
        "percentage_error": pct_err, 
        "inference_time_sec": t,
    }

def average_metrics(mlist: list) -> dict:
    """
    Menghitung rata-rata dari daftar metrik.
    Hanya memproses IoU, Dice Coefficient, Percentage Error, dan Waktu Inferensi.
    """
    # Pemetaan dari key input (dari compute_all_metrics) ke key output rata-rata
    target_metrics = {
        "iou": "iou",
        "dice": "dice_coefficient",  # Mengubah nama menjadi Dice Coefficient sesuai permintaan
        "percentage_error": "percentage_error",
        "inference_time_sec": "inference_time_sec"
    }
    
    result = {}
    
    for input_key, output_key in target_metrics.items():
        # Ambil nilai metrik yang tersedia dari daftar data mlist
        vals = [m[input_key] for m in mlist if m.get(input_key) is not None]
        # Hitung rata-rata jika datanya ada
        result[output_key] = float(np.mean(vals)) if vals else None
        
    return result