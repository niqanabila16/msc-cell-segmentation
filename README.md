<div align="center">
  <h1>🔬 APE MSC</h1>
  <p><b>Aplikasi Pendukung Eksperimen</b> for evaluating U-Net-based MSC segmentation models</p>
</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
</div>

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [About APE MSC](#about-ape-msc)
3. [Usage — Inference Flow](#usage--inference-flow)
4. [App Screenshots](#app-screenshots)
5. [Research Results](#research-results)
6. [Overlay Colors](#overlay-colors)
7. [Contributors](#contributors)
8. [Technologies We Use](#technologies-we-use)
9. [Project Stats](#project-stats)
10. [Setup](#setup)
11. [Installation (Local)](#installation-local)
12. [License](#license)

---

<div align="center">

<h2 id="architecture-overview">Architecture Overview</h2>

### Experiment Process Flow


![APE MSC Flowchart](/ape_msc/assets/alur_proses_ape.jpg)


![Eksperimen penelitian Flowchart](/ape_msc/assets/alur_proses_eksperimen.jpg)


</div>

---

<h2 id="about-ape-msc">About APE MSC</h2>

APE (Aplikasi Pendukung Eksperimen) is a Streamlit application built to support MSC segmentation experiments. It has one main documented feature: run U-Net inference on microscopy images — with or without ground truth labels — and display accuracy metrics. *(Bab IV.9.1)*

Functional scope, per Tabel IV.3 (FR01–FR10):
- Accept one or more microscopy images plus optional ground truth labels through the interface
- Automatically match each image to its label by filename
- Validate image (`.jpg/.jpeg/.png`) and label (`.png`) files, max 10 MB each
- Run U-Net inference, with or without ground truth
- Compute IoU, Dice Coefficient, Precision, Recall, and confluence percentage when ground truth is available
- Display the original image, ground truth (if any), prediction, and a colored overlay mask
- Adjustable overlay transparency
- Dropdown selection of image preprocessing method

> Only this single inference feature is documented in the thesis — no separate "model comparison" tab is described there.

---

<h2 id="usage--inference-flow">Usage — Inference Flow</h2>

Per Bab IV.9.2:

1. Upload one or more microscopy images — required, `.jpg/.jpeg/.png`, max 10 MB each. Upload matching ground truth labels — optional, `.png`, max 10 MB. Labels are matched automatically by filename (same name, or same name with a `_mask` suffix).
2. Choose an image preprocessing option from the dropdown (optional).
3. Adjust overlay transparency as needed.
4. Run inference — each image is processed individually through the U-Net model; ground truth (if present) does not affect the prediction itself.
5. Results shown per image:
   - **With GT:** Original | GT | Prediction | Overlay (TP green, FP yellow, FN red)
   - **Without GT:** Original | Prediction | Overlay (blue = predicted area)
   - Metrics (when GT is available): IoU, Dice Coefficient, Precision, Recall, actual & predicted confluence, inference time
6. After all images are processed, an average-metrics summary is shown, with results available to export/download.

---

<h2 id="app-screenshots">App Screenshots</h2>

<div align="center">

*Main dashboard*

![APE MSC Dashboard](/ape_msc/assets/dashboard.png)

*Inference result with a ground truth label — Original \| GT \| Prediction \| TP/FP/FN Overlay*

![Segmentation result with GT](/ape_msc/assets/hasil_segmentasi_citra_GT.png)

*Inference result without a label — Original \| Prediction \| Overlay*

![Segmentation result without GT](/ape_msc/assets/hasil_segmentasi_citra_noGT.png)

</div>
---

<h2 id="research-results">Research Results</h2>

Source: `_Laporan_TA_final.pdf`, Bab V. Two datasets: **phase-contrast** (main data) and **non-phase-contrast** (generalization test).

### Phase-Contrast — Best Results

| Configuration | Dice | IoU |
|---|---|---|
| Baseline U-Net | 0.8816 | 0.7909 |
| + Best augmentation (Gaussian Blur) | 0.8885 | 0.8020 |
| + Best encoder (ResNet50, paper config) | 0.8901 | 0.8041 |
| + Best optimizer (AdamN, paper config) | 0.8951 | 0.8128 |
| + Best activation function (AFpM, paper config) | 0.8961 | 0.8141 |
| **Final Model (Decoder)** — all four combined | **0.8988** | **0.8185** |

**Final Model (Decoder)** is the best result overall on phase-contrast — it beats the baseline and every individual ablation.

### Non-Phase-Contrast — Best Results

| Configuration | Dice | IoU |
|---|---|---|
| Baseline U-Net | 0.0039 | 0.0020 |
| Final Model (Decoder), no preprocessing | 0.0626 | 0.0330 |
| **Final Model (Decoder) + Polynomial Background Correction** | **0.5726** | **0.4254** |

Preprocessing that reshapes non-phase-contrast images to look more like the phase-contrast training data is what drives generalization — Polynomial Background Correction gives the best non-phase-contrast result.

<div align="center">

*Segmentation result of Final model Decoder — Original \| GT \| Prediction U-Net \| Prediction Final Model Decoder*

![Segmentation result of Final model Decoder](/ape_msc/assets/table_final_model.png)

*Segmentation result of Final model Decoder — Original \| GT \| Prediction U-Net \| Prediction Final Model Decoder*

![Segmentation result of Final model Decoder + pre-processing](/ape_msc/assets/table_preprocessing_final_model.png)

</div>

`[IMAGE: Table V.6 — Final Model results, p.197]` → save as `assets/table_final_model.png`
`[IMAGE: Table V.7 — Preprocessing on Final Model, p.200]` → save as `assets/table_preprocessing_final_model.png`

### Ablation Notes

- **Augmentation:** on non-phase-contrast, only lighting-related techniques (low-resolution, gamma, brightness) helped — flip/blur/noise didn't.
- **Encoder & optimizer:** only beat baseline when paired with their paper-recommended config; default hyperparameters weren't enough.

---

<h2 id="overlay-colors">Overlay Colors</h2>

Confirmed by FR08 and Bab IV.9.2 step 7:

| Color | Meaning |
|-------|-------|
| 🟢 Green  | **TP** — True Positive (cell correctly predicted) |
| 🟡 Yellow | **FP** — False Positive (incorrectly predicted as cell) |
| 🔴 Red    | **FN** — False Negative (cell not detected) |
| 🔵 Blue   | **Prediction** area shown when no label is provided |

---

<h2 id="contributors" align="center">🌟 Contributors 🌟</h2>
<div align="center">
  <table border="0">
    <tr>
      <td align="center">
        <a href="https://github.com/niqanabila16">
          <img src="https://avatars.githubusercontent.com/niqanabila16" width="100" alt="niqanabila16" style="border-radius: 50%;" />
        </a>
        <br>
        <a href="https://github.com/niqanabila16">niqanabila16</a>
      </td>
      <td align="center">
        <a href="https://github.com/asrihusnull">
          <img src="https://avatars.githubusercontent.com/asrihusnull" width="100" alt="asrihusnull" style="border-radius: 50%;" />
        </a>
        <br>
        <a href="https://github.com/asrihusnull">asrihusnull</a>
      </td>
    </tr>
  </table>
</div>

---

<h2 id="technologies-we-use" align="center">🚀 Technologies We Use 🚀</h2>
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/segmentation--models--pytorch-EE4C2C?style=for-the-badge" alt="segmentation-models-pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/scikit--image-000000?style=for-the-badge" alt="scikit-image" />
  <img src="https://img.shields.io/badge/Albumentations-EE4C2C?style=for-the-badge" alt="Albumentations" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge" alt="Seaborn" />
</div>
<p align="center"><i>Per Tabel IV.4 (PyTorch 2.10.0, OpenCV 4.13.0, scikit-image 0.25.2, Albumentations 2.0.8, NumPy 2.0.2, scikit-learn 1.8.0, Matplotlib 3.10.0, Seaborn 0.13.2)</i></p>

---

<h2 id="project-stats" align="center">📈 Project Stats 📈</h2>
<div align="center">
  <img src="https://img.shields.io/github/last-commit/niqanabila16/msc-cell-segmentation?color=yellow" alt="Last Commit" />
  <img src="https://img.shields.io/github/stars/niqanabila16/msc-cell-segmentation?color=blue" alt="Stars" />
</div>

---

<h2 id="setup">Setup</h2>

```powershell
git clone https://github.com/niqanabila16/msc-cell-segmentation.git
cd msc-cell-segmentation/ape_msc
```

Requires Python 3.10.11.

---

<h2 id="installation-local">Installation (Local)</h2>

```powershell
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
streamlit run app.py
```

Open in browser: http://localhost:8501

---

<h2 id="license">License</h2>

This project was developed as a Tugas Akhir (thesis) at Politeknik Negeri Bandung (Polban).