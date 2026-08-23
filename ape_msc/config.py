"""
config.py — APE MSC.
Auto-scan models/, override via .env jika diperlukan.

Perubahan v4:
- SAM2 dihapus sepenuhnya: _discover_sam2, SAM2_VERSIONS, SAM2_INPUT_SIZE,
  parameter SAM2AutomaticMaskGenerator, dan direktori models/sam2. Aplikasi
  sekarang hanya menjalankan U-Net (vanilla atau SMP ResNet50).

Perubahan dari v3:
- Tambah POLY3_ORDER, PERCENTILE_LOW, PERCENTILE_HIGH
  untuk pipeline preprocessing berbasis luminance.
"""

import json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR: Path = Path(__file__).parent.resolve()

# ── Direktori ──────────────────────────────────────────────────────────────────
OUTPUT_DIR     = Path(os.getenv("OUTPUT_DIR",  str(BASE_DIR / "outputs")))
INFERENCE_DIR  = OUTPUT_DIR / "inference"
COMPARISON_DIR = OUTPUT_DIR / "comparisons"
TEMP_DIR       = Path(os.getenv("TEMP_DIR",    str(BASE_DIR / "temp")))

# ── Auto-discover checkpoints ──────────────────────────────────────────────────
def _discover_unet() -> list:
    folder = BASE_DIR / "models" / "unet"
    out = []
    if folder.exists():
        for f in sorted(folder.glob("*.pth")) + sorted(folder.glob("*.pt")):
            out.append({"label": f.name, "path": str(f)})
    if not out:
        out.append({
            "label": "unet_msc.pth",
            "path":  str(BASE_DIR / "models" / "unet" / "unet_msc.pth"),
        })
    return out


_UNET_ENV = os.getenv("UNET_VERSIONS", "")
UNET_VERSIONS: list = json.loads(_UNET_ENV) if _UNET_ENV else _discover_unet()

# ── Ukuran input model ─────────────────────────────────────────────────────────
UNET_INPUT_SIZE = (512,  512)    # (W, H)

# ── Validasi upload ────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB    = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_IMAGE_EXTS  = (".jpg", ".jpeg", ".png")
ALLOWED_LABEL_EXTS  = (".png", ".json")
MIN_IMAGE_DIM       = int(os.getenv("MIN_IMAGE_DIM", "64"))

# ── Preprocessing — CLAHE (LAB) ────────────────────────────────────────────────
CLAHE_CLIP_LIMIT = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))
CLAHE_TILE_GRID  = int(os.getenv("CLAHE_TILE_GRID",    "8"))

# ── Preprocessing — Pipeline luminance ────────────────────────────────────────
# Orde polinomial untuk koreksi background (poly3).
# Nilai 3 sudah cukup untuk menangkap gradien iluminasi non-linear pada MSC.
POLY3_ORDER      = int(os.getenv("POLY3_ORDER",      "3"))

# Batas bawah dan atas persentil untuk percentile rescale.
# Default 1.0–99.0 memotong 1 % outlier di tiap ujung.
PERCENTILE_LOW   = float(os.getenv("PERCENTILE_LOW",  "1.0"))
PERCENTILE_HIGH  = float(os.getenv("PERCENTILE_HIGH", "99.0"))

# ── Visualisasi ───────────────────────────────────────────────────────────────
OVERLAY_ALPHA = float(os.getenv("OVERLAY_ALPHA", "0.55"))
COLOR_TP   = (0,   210,   0)    # Hijau
COLOR_FP   = (255, 220,   0)    # Kuning
COLOR_FN   = (220,   0,   0)    # Merah
COLOR_PRED = (0,   120, 255)    # Biru

# ── Misc ──────────────────────────────────────────────────────────────────────
EPSILON   = 1e-7
APP_TITLE = "APE — Aplikasi Pendukung Eksperimen MSC"
APP_ICON  = ""

# Auto-create dirs
for _d in [INFERENCE_DIR, COMPARISON_DIR, TEMP_DIR, BASE_DIR / "models" / "unet"]:
    Path(_d).mkdir(parents=True, exist_ok=True)