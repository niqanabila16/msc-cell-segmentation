# MSC Cell Segmentation using U-Net, SAM2, and APE

This repository contains experiments and implementations for **mesenchymal stem cell (MSC) segmentation** using multiple approaches, with a focus on:

- High segmentation accuracy
- Generalization across different microscopy environments (brightfield, phase contrast)
- Efficient inference on low-spec devices (CPU-based)

---

## 📌 Project Overview

Microscopy images of MSC often suffer from:
- Low contrast
- Noise
- Variability across imaging devices

This project explores how **preprocessing techniques** and **model architecture design** (U-Net, SAM2, APE) can improve segmentation performance and robustness.

## Status

- Fine-tuned project: coming soon
- U-Net directory: coming soon
- Training notebook: coming soon

## Related Training Notebook

Training notebook for SAM2:
[Kaggle Notebook](https://www.kaggle.com/code/niqanabila/mesenchymal-stem-cell-sam-2)

Training notebook for U-Net: coming soon

## Notes

This folder follows the original SAM2 repository structure from the official source.

---

## 📁 Repository Structure

```text
msc-cell-segmentation/
│
├── ape_msc/        # APE-based application (Streamlit)
├── sam2/           # Official SAM2 repository (cloned)
├── unet/           # U-Net implementation (coming soon)
│
├── data/           # (ignored) dataset
├── outputs/        # (ignored) results, predictions
│
├── README.md
└── requirements.txt (optional)

