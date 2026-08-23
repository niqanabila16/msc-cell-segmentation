"""
diagnose_model.py — Jalankan TERPISAH dari Streamlit untuk verifikasi model_loader.

Cara pakai:
  1. Sesuaikan CHECKPOINT_PATH di bawah
  2. Jalankan dari terminal: python diagnose_model.py
  3. Baca output — tidak boleh ada error, _smp harus True untuk ResNet50

Tidak perlu Streamlit, tidak ada cache.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# ── Sesuaikan path di bawah ─────────────────────────────────────────────────
CHECKPOINT_PATH = r"D:\Eksperimen TA\msc-cell-segmentation\Pytorch-UNet\model\Resnet50.pt"
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 60


def banner(title): print(f"\n{SEP}\n  {title}\n{SEP}")


def main():
    banner("1. Verifikasi file checkpoint")
    p = Path(CHECKPOINT_PATH)
    if not p.exists():
        print(f"[ERROR] File tidak ditemukan: {p}")
        sys.exit(1)
    print(f"[OK] File ditemukan: {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    banner("2. Baca checkpoint mentah")
    device = torch.device("cpu")
    try:
        ckpt = torch.load(str(p), map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(str(p), map_location=device)

    if isinstance(ckpt, dict):
        print(f"[OK] Format: dict  |  Top-level keys: {list(ckpt.keys())}")
        ckpt_cfg = ckpt.get("config", {})
        print(f"     config  : {ckpt_cfg}")
        print(f"     epoch   : {ckpt.get('epoch')}")
        print(f"     val_dice: {ckpt.get('best_val_dice')}")
    else:
        print(f"[WARN] Format bukan dict: {type(ckpt)}")

    banner("3. Ekstrak dan bersihkan state_dict")
    # Ikuti urutan _load_checkpoint_and_extract
    raw_sd = None
    for key in ("model_state_dict", "state_dict", "model", "net"):
        if key in ckpt and isinstance(ckpt[key], dict):
            raw_sd = ckpt[key]
            print(f"[OK] state_dict ditemukan di key: '{key}'")
            break
    if raw_sd is None:
        raw_sd = ckpt
        print("[WARN] Menggunakan root dict sebagai state_dict")

    state_dict = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in raw_sd.items()
        if isinstance(v, torch.Tensor)
    }
    print(f"[OK] Jumlah tensor: {len(state_dict)}")

    banner("4. Sampel kunci state_dict (penting untuk deteksi arsitektur)")
    sample = list(state_dict.keys())[:15]
    for k in sample:
        print(f"     {k}  →  {tuple(state_dict[k].shape)}")

    banner("5. Deteksi jenis arsitektur")
    SMP_PREFIXES = {"encoder.", "decoder.", "segmentation_head."}
    VANILLA_PREFIXES = {"inc.", "down1.", "up1.", "outc."}
    is_smp = any(
        any(k.startswith(p) for p in SMP_PREFIXES)
        for k in sample
    )
    is_vanilla = any(
        any(k.startswith(p) for p in VANILLA_PREFIXES)
        for k in sample
    )
    print(f"     is_smp    : {is_smp}   (diharapkan: True untuk ResNet50-UNet)")
    print(f"     is_vanilla: {is_vanilla}   (diharapkan: False untuk ResNet50-UNet)")

    if is_smp:
        print("[OK] Checkpoint akan dimuat sebagai SMP U-Net")
    elif is_vanilla:
        print("[WARN] Checkpoint terdeteksi sebagai vanilla U-Net — pastikan ini benar")
    else:
        print("[ERROR] Prefix tidak dikenal — periksa key di atas secara manual")

    banner("6. Full load via load_unet()")
    try:
        from src.model_loader import load_unet
    except ImportError:
        print("[ERROR] Tidak bisa import src.model_loader.")
        print("         Jalankan script ini dari root folder ape_msc/, bukan dari src/")
        sys.exit(1)

    try:
        model = load_unet(CHECKPOINT_PATH)
    except RuntimeError as e:
        print(f"[ERROR] load_unet() gagal:\n{e}")
        sys.exit(1)

    print(f"[OK] load_unet() berhasil")
    print(f"     type(model)   : {type(model).__name__}")
    print(f"     model._smp    : {getattr(model, '_smp', False)}")
    print(f"     model._classes: {getattr(model, '_smp_classes', 'N/A')}")

    banner("7. Forward pass dummy (CPU, gambar 1024x1024)")
    from src.image_utils import image_to_tensor_imagenet, image_to_tensor

    dummy = np.zeros((1024, 1024, 3), dtype=np.uint8)
    is_smp_model = getattr(model, '_smp', False)

    if is_smp_model:
        tensor = image_to_tensor_imagenet(dummy)
        print(f"[OK] Tensor dibuat dengan ImageNet norm: {tuple(tensor.shape)}")
        print(f"     min={tensor.min():.4f}  max={tensor.max():.4f}  mean={tensor.mean():.4f}")
        expected_min = round((-0.485 / 0.229), 2)
        print(f"     (pixel [0,0,0] → R=({0/255.0:.3f}-0.485)/0.229 ≈ {expected_min:.2f}  — sesuai?)")
    else:
        tensor = image_to_tensor(dummy)
        print(f"[OK] Tensor dibuat /255.0: {tuple(tensor.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(tensor)

    print(f"[OK] Forward pass berhasil")
    print(f"     output shape: {tuple(output.shape)}")

    if is_smp_model:
        expected_shape = (1, 1, 1024, 1024)
        if tuple(output.shape) == expected_shape:
            print(f"[OK] Shape sesuai ekspektasi SMP: {expected_shape}")
        else:
            print(f"[WARN] Shape tidak sesuai ekspektasi {expected_shape}")
        pred = (torch.sigmoid(output) > 0.5).long()
        print(f"     Unique pred values: {pred.unique().tolist()}")
    else:
        expected_shape = (1, 2, 512, 512)
        print(f"[INFO] Vanilla UNet output shape: {tuple(output.shape)}")
        pred = torch.argmax(output, dim=1)
        print(f"     Unique pred values: {pred.unique().tolist()}")

    banner("HASIL AKHIR")
    if is_smp_model and tuple(output.shape)[1] == 1:
        print("[PASS] Semua cek lulus. model_loader.py bekerja dengan benar.")
        print("       Jika app.py masih salah → bersihkan Streamlit cache (lihat README).")
    else:
        print("[FAIL] Ada yang tidak sesuai — periksa output di atas.")


if __name__ == "__main__":
    main()
