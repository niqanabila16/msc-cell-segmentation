import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

_unet_cache: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# KOMPATIBILITAS TRAINING LOKAL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CheckpointProfile:
    """Ringkasan metadata checkpoint agar inferensi bisa meniru lokal."""
    activation: str = "relu"
    in_channels: int = 3
    base_features: int = 64
    num_classes: int = 2
    scale: float = 0.5
    mask_values: tuple[int, ...] = (0, 1)
    is_smp: bool = False
    encoder_name: str = "resnet50"


def preprocess_image_like_local(image: Image.Image | np.ndarray, scale: float = 0.5, in_channels: int = 3) -> np.ndarray:
    """
    Preprocess citra mirip BasicDataset.preprocess(..., is_mask=False) dari training lokal.

    in_channels menentukan mode warna: 1 = grayscale ("L"), selain itu RGB.
    Output: float32 CHW dengan nilai [0..1].
    """
    mode = "L" if in_channels == 1 else "RGB"

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = Image.fromarray(image).convert(mode)
        else:
            image = Image.fromarray(image.astype(np.uint8)).convert(mode)
    else:
        image = image.convert(mode)

    if scale != 1.0:
        new_w = max(1, int(round(image.width * scale)))
        new_h = max(1, int(round(image.height * scale)))
        image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    if in_channels == 1:
        arr = arr[np.newaxis, :, :]  # (1, H, W)
    else:
        arr = np.transpose(arr, (2, 0, 1))  # (3, H, W)
    return np.ascontiguousarray(arr)


def preprocess_mask_like_local(mask: Image.Image | np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Preprocess mask mirip BasicDataset.preprocess(..., is_mask=True) dari training lokal.
    """
    if isinstance(mask, np.ndarray):
        if mask.ndim == 3:
            mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
        else:
            mask = Image.fromarray(mask.astype(np.uint8)).convert("L")
    else:
        mask = mask.convert("L")

    if scale != 1.0:
        new_w = max(1, int(round(mask.width * scale)))
        new_h = max(1, int(round(mask.height * scale)))
        mask = mask.resize((new_w, new_h), resample=Image.NEAREST)

    arr = np.asarray(mask, dtype=np.int64)
    return np.ascontiguousarray(arr)


# ══════════════════════════════════════════════════════════════════════════════
# FUNGSI AKTIVASI KUSTOM
# ══════════════════════════════════════════════════════════════════════════════

class AFpM(nn.Module):
    """Adaptive Flatten p-Mish (AFpM).

    Mendukung DUA cara penyimpanan parameter `p`, supaya kompatibel dengan
    checkpoint LAMA maupun BARU:

      - Checkpoint LAMA (p_bank=None): tiap instance AFpM punya
        nn.Parameter(1,) sendiri-sendiri (versi awal, sebelum fix bug CUDA
        "misaligned address" pada nn.DataParallel).
      - Checkpoint BARU (p_bank diisi): seluruh instance AFpM dalam SATU
        scope (encoder ATAU decoder) berbagi SATU tensor tied `p_bank`,
        tiap instance mengambil elemen ke-`idx` miliknya sendiri. Ini pola
        yang sekarang dipakai notebook training (root-cause fix untuk bug
        CUDA "misaligned address" nn.DataParallel + banyak parameter kecil).
    """

    def __init__(self, p_bank: Optional[nn.Parameter] = None, idx: int = 0):
        super().__init__()
        if p_bank is not None:
            self.idx = idx
            self.p = p_bank          # tied/shared Parameter (weight tying)
        else:
            self.idx = None
            self.p = nn.Parameter(torch.empty(1))
            nn.init.xavier_normal_(self.p.unsqueeze(0))

    def forward(self, z):
        z = z.contiguous()
        p_scalar = self.p[self.idx] if self.idx is not None else self.p
        p = p_scalar.reshape(1, 1, 1, 1)
        mish = z * torch.tanh(F.softplus(z))
        return torch.where(z >= 0, mish + p, p)


class SbPiPLU(nn.Module):
    """Softsign-based Piecewise Parametric Linear Unit (scalar k)."""

    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(21.0))

    def forward(self, z):
        z = z.contiguous()
        k_val = self.k.view(1, 1, 1, 1)
        s_x = F.softsign(z)
        neg_part = 2 * s_x + 0.5 * (s_x ** 2)
        return torch.where(z <= 0, neg_part, torch.where(z <= k_val, z, z / k_val))


# ══════════════════════════════════════════════════════════════════════════════
# VANILLA U-Net (kustom)
# ══════════════════════════════════════════════════════════════════════════════

def get_activation(act_name: str) -> nn.Module:
    act_name = act_name.lower()
    if "afpm" in act_name:
        return AFpM()
    if "sbpiplu" in act_name:
        return SbPiPLU()
    return nn.ReLU(inplace=True)


class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, act_name="relu"):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(act_name),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            get_activation(act_name),
        )

    def forward(self, x):
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_ch, out_ch, act_name="relu"):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_ch, out_ch, act_name))

    def forward(self, x):
        return self.maxpool_conv(x)


class _Up(nn.Module):
    def __init__(self, in_ch, out_ch, act_name="relu"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch, act_name)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_features=64, act_name="relu"):
        super().__init__()
        f = base_features
        self.inc = _DoubleConv(in_channels, f, act_name)
        self.down1 = _Down(f, f * 2, act_name)
        self.down2 = _Down(f * 2, f * 4, act_name)
        self.down3 = _Down(f * 4, f * 8, act_name)
        self.down4 = _Down(f * 8, f * 16, act_name)
        self.up1 = _Up(f * 16, f * 8, act_name)
        self.up2 = _Up(f * 8, f * 4, act_name)
        self.up3 = _Up(f * 4, f * 2, act_name)
        self.up4 = _Up(f * 2, f, act_name)
        self.outc = _OutConv(f, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ══════════════════════════════════════════════════════════════════════════════
# DETEKSI JENIS CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

_SMP_PREFIXES = {"encoder.", "decoder.", "segmentation_head."}


def _is_smp_checkpoint(state_dict: dict) -> bool:
    sample = list(state_dict.keys())[:30]
    return any(any(k.startswith(p) for p in _SMP_PREFIXES) for k in sample)


def _detect_vanilla_activation(state_dict: dict, ckpt_name: str, ckpt_cfg: dict | None = None) -> str:
    ckpt_cfg = ckpt_cfg or {}
    act_cfg = str(ckpt_cfg.get("activation", "")).lower()
    if "afpm" in act_cfg:
        return "afpm"
    if "sbpiplu" in act_cfg:
        return "sbpiplu"

    has_p = any(k.endswith(".p") for k in state_dict.keys())
    has_k = any(k.endswith(".k") for k in state_dict.keys())
    if has_p and not has_k:
        return "afpm"
    if has_k and not has_p:
        return "sbpiplu"

    name_lower = ckpt_name.lower()
    if "afpm" in name_lower:
        return "afpm"
    if "sbpiplu" in name_lower:
        return "sbpiplu"
    return "relu"


def _detect_in_channels(state_dict: dict, ckpt_cfg: dict, meta: dict) -> int:
    """
    Deteksi jumlah channel input model dari (urutan prioritas):
      1. ckpt_cfg["in_channels"] / meta["in_channels"] jika tersedia
      2. shape aktual weight conv pertama ('inc.double_conv.0.weight') di checkpoint
      3. fallback default 3 (RGB)
    Poin 2 penting karena checkpoint hasil train.py (milesial) tidak menyimpan
    'config' terpisah, sehingga in_channels harus disimpulkan dari shape weight.
    """
    if "in_channels" in ckpt_cfg:
        return int(ckpt_cfg["in_channels"])
    if "in_channels" in meta:
        return int(meta["in_channels"])

    first_conv_key = "inc.double_conv.0.weight"
    if first_conv_key in state_dict:
        return int(state_dict[first_conv_key].shape[1])

    return 3


# ══════════════════════════════════════════════════════════════════════════════
# PEMBACA CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_mask_values(value: Any) -> list[int]:
    if value is None:
        return [0, 1]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out or [0, 1]
    try:
        return [int(value)]
    except Exception:
        return [0, 1]


def _load_checkpoint_and_extract(weights_path: Path, device: torch.device) -> tuple[dict, dict, dict]:
    try:
        try:
            ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(weights_path), map_location=device)
    except Exception as e:
        raise RuntimeError(f"Gagal membaca checkpoint '{weights_path}': {e}") from e

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Format checkpoint tidak dikenal: {type(ckpt)}. Diharapkan dict dengan key 'model_state_dict' atau state_dict langsung."
        )

    ckpt_cfg: dict = {}
    meta: dict = {}
    raw_sd: dict | None = None

    for meta_key in ("mask_values", "config", "scale", "input_scale", "in_channels", "num_classes", "activation", "base_features"):
        if meta_key in ckpt:
            meta[meta_key] = ckpt[meta_key]

    for key in ("model_state_dict", "state_dict", "model", "net"):
        if key in ckpt and isinstance(ckpt[key], dict):
            raw_sd = ckpt[key]
            if key == "model_state_dict":
                ckpt_cfg = ckpt.get("config", {}) or {}
                if not isinstance(ckpt_cfg, dict):
                    ckpt_cfg = {}
            elif key == "state_dict" and isinstance(ckpt.get("config"), dict):
                ckpt_cfg = ckpt.get("config", {}) or {}
            break

    if raw_sd is None:
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            raw_sd = ckpt
        else:
            raise ValueError(f"Tidak bisa menemukan state_dict dalam checkpoint. Keys yang tersedia: {list(ckpt.keys())}")

    if isinstance(ckpt.get("config"), dict) and not ckpt_cfg:
        ckpt_cfg = ckpt.get("config", {}) or {}

    state_dict = {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in raw_sd.items()
        if isinstance(v, torch.Tensor)
    }

    if not state_dict:
        raise ValueError(f"Tidak ada tensor dalam state_dict. Keys ditemukan: {list(raw_sd.keys())[:10]}")

    if "mask_values" in meta:
        meta["mask_values"] = _normalize_mask_values(meta["mask_values"])

    return state_dict, ckpt_cfg, meta


def _adapt_state_dict_shapes(state_dict: dict, model: nn.Module) -> dict:
    model_sd = model.state_dict()
    adapted = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key in model_sd:
            ref = model_sd[key]
            if value.shape != ref.shape and value.numel() == ref.numel():
                try:
                    value = value.reshape(ref.shape)
                    logger.warning(f"Menyesuaikan shape '{key}': {tuple(state_dict[key].shape)} -> {tuple(ref.shape)}")
                except Exception:
                    pass
        adapted[key] = value
    return adapted


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER: SMP ResNet50-UNet
# ══════════════════════════════════════════════════════════════════════════════

def _detect_smp_activation(state_dict: dict, scope: str) -> str:
    """
    Deteksi aktivasi kustom (AFpM/SbPiPLU) yang dipakai di bagian checkpoint
    SMP dengan prefix key tertentu ('decoder.' atau 'encoder.'). Notebook
    ablation AFpM bisa menempatkan AFpM di decoder saja, encoder saja, atau
    keduanya -- jadi decoder dan encoder dideteksi terpisah.
    """
    has_p = any(k.startswith(scope) and k.endswith(".p") for k in state_dict.keys())
    has_k = any(k.startswith(scope) and k.endswith(".k") for k in state_dict.keys())
    if has_p and not has_k:
        return "afpm"
    if has_k and not has_p:
        return "sbpiplu"
    return "relu"


def _detect_afpm_bank_size(state_dict: dict, scope: str) -> Optional[int]:
    """
    Deteksi ukuran tied-parameter-bank AFpM untuk scope tertentu ('encoder.'
    atau 'decoder.') langsung dari checkpoint -- supaya loader otomatis
    kompatibel baik dengan checkpoint LAMA maupun BARU tanpa perlu tahu versi
    notebook mana yang dipakai untuk training.

    Checkpoint LAMA (sebelum fix bug CUDA "misaligned address" nn.DataParallel
    + AFpM di notebook training): tiap instance AFpM py nn.Parameter(1,)
    independen -> tiap key '.p' pada scope itu berukuran (1,).
    Checkpoint BARU (setelah fix): SEMUA instance AFpM pada scope yang sama
    berbagi SATU tensor tied -> tiap key '.p' pada scope itu berukuran SAMA,
    (n_padded,) dengan n_padded > 1 (nilainya pun identik di semua key,
    karena memang tensor yang sama -- lihat catatan di notebook training).

    Return None kalau tidak ada key '.p' sama sekali pada scope ini.
    """
    shapes = {
        tuple(v.shape)
        for k, v in state_dict.items()
        if k.startswith(scope) and k.endswith(".p") and v.dim() == 1
    }
    if not shapes:
        return None
    return max(s[0] for s in shapes)


def _patch_smp_decoder_activation(model, act_name: str, ckpt_name: str, bank_size: Optional[int] = None) -> None:
    """
    Ganti slot aktivasi (indeks ke-2 di Conv2dReLU -- Sequential[conv, bn, act])
    tiap decoder block SMP dengan AFpM/SbPiPLU, WAJIB dipanggil SEBELUM
    load_state_dict() supaya parameter '.p'/'.k' di checkpoint punya modul
    tujuan yang cocok (kalau tidak: RuntimeError size/unexpected/missing keys).

    bank_size: kalau act_name == "afpm" dan checkpoint memakai tied-parameter-
    bank (checkpoint BARU, hasil notebook training setelah fix CUDA
    "misaligned address"), diisi ukuran bank yang terdeteksi lewat
    _detect_afpm_bank_size(). SEMUA instance AFpM decoder lalu berbagi SATU
    nn.Parameter(bank_size,) ini (bukan Parameter(1,) sendiri-sendiri) --
    urutan iterasi di bawah (block demi block, conv1 lalu conv2) SENGAJA
    disamakan persis dengan urutan penomoran indeks di notebook training
    (rekursi named_children(), conv1 sebelum conv2 tiap block, block
    berurutan) supaya index-nya konsisten.
    """
    if act_name == "relu":
        return

    p_bank = None
    if act_name == "afpm" and bank_size and bank_size > 1:
        p_bank = nn.Parameter(torch.zeros(bank_size))

    act_idx = 0
    n_patched = 0
    for block in model.decoder.blocks:
        for conv_attr in ("conv1", "conv2"):
            seq = getattr(block, conv_attr, None)
            if isinstance(seq, nn.Sequential) and len(seq) >= 3:
                if act_name == "afpm":
                    seq[2] = AFpM(p_bank, act_idx)
                    act_idx += 1
                else:
                    seq[2] = SbPiPLU()
                n_patched += 1
    bank_msg = f", tied bank shape ({bank_size},)" if p_bank is not None else ""
    logger.info(f"SMP U-Net '{ckpt_name}': decoder activation di-patch ke {act_name.upper()} ({n_patched} modul{bank_msg})")


def _patch_smp_encoder_activation(model, act_name: str, ckpt_name: str, bank_size: Optional[int] = None) -> None:
    """
    Ganti nn.ReLU di stem + tiap Bottleneck block encoder ResNet50 dengan
    AFpM/SbPiPLU. Torchvision Bottleneck memakai SATU modul '.relu' yang
    dipanggil ulang 3x per forward, jadi hanya ada satu '.p'/'.k' per block.

    bank_size: sama seperti di _patch_smp_decoder_activation -- kalau diisi
    (>1), seluruh instance AFpM ENCODER berbagi SATU nn.Parameter(bank_size,)
    (tied), bukan Parameter(1,) sendiri-sendiri. Urutan iterasi di bawah
    (stem dulu, lalu layer1..layer4 block demi block) SENGAJA disamakan
    persis dengan urutan penomoran indeks di notebook training.
    """
    if act_name == "relu":
        return

    p_bank = None
    if act_name == "afpm" and bank_size and bank_size > 1:
        p_bank = nn.Parameter(torch.zeros(bank_size))

    act_idx_holder = {"i": 0}

    def _make_act():
        if act_name == "afpm":
            i = act_idx_holder["i"]
            act_idx_holder["i"] += 1
            return AFpM(p_bank, i)
        return SbPiPLU()

    encoder = model.encoder
    n_patched = 0
    if hasattr(encoder, "relu"):
        encoder.relu = _make_act()
        n_patched += 1
    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(encoder, layer_name, None)
        if layer is None:
            continue
        for block in layer:
            if hasattr(block, "relu"):
                block.relu = _make_act()
                n_patched += 1
    bank_msg = f", tied bank shape ({bank_size},)" if p_bank is not None else ""
    logger.info(f"SMP U-Net '{ckpt_name}': encoder activation di-patch ke {act_name.upper()} ({n_patched} modul{bank_msg})")


def _build_smp_model(state_dict: dict, ckpt_cfg: dict, ckpt_name: str, device: torch.device) -> torch.nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            f"Gagal mengimport segmentation_models_pytorch: {e} Kemungkinan dependency yang hilang. Coba: pip install six pip install segmentation-models-pytorch"
        ) from e

    encoder_name = ckpt_cfg.get("encoder_name", "resnet50")
    in_channels = int(ckpt_cfg.get("in_channels", 3))
    task_type = str(ckpt_cfg.get("task_type", "binary")).lower()
    num_classes = 1 if task_type == "binary" else int(ckpt_cfg.get("num_classes", 1))

    # Resolusi training (kalau tersimpan di checkpoint config). SegmentationDataset
    # training resize setiap citra ke ukuran ABSOLUT input_h x input_w (bukan
    # proporsional per-citra) sebelum masuk model -- ini WAJIB direplikasi saat
    # inferensi, kalau tidak skala objek (sel) relatif ke receptive field
    # ResNet50 akan berbeda dari training dan hasil segmentasi jadi berbercak.
    input_h = ckpt_cfg.get("input_h")
    input_w = ckpt_cfg.get("input_w")
    scale = ckpt_cfg.get("scale")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )

    # Deteksi & pasang aktivasi kustom (AFpM/SbPiPLU) SEBELUM load_state_dict --
    # checkpoint dari notebook ablation AFpM (afpm_decoder / afpm_encoder_decoder)
    # menyimpan modul aktivasi kustom, bukan ReLU polos bawaan smp.Unet. Tanpa
    # patch ini, load_state_dict(strict=True) gagal dengan "Unexpected key(s)
    # ...conv1.2.p / conv2.2.p" karena Sequential[2] di model masih nn.ReLU
    # (tidak punya parameter), sedangkan checkpoint punya parameter '.p' di sana.
    decoder_act = _detect_smp_activation(state_dict, "decoder.")
    encoder_act = _detect_smp_activation(state_dict, "encoder.")

    # Deteksi ukuran tied-parameter-bank AFpM langsung dari checkpoint (kalau
    # ada) -- supaya kompatibel baik dengan checkpoint LAMA (Parameter(1,)
    # per instance) maupun checkpoint BARU hasil notebook training yang sudah
    # di-fix untuk bug CUDA "misaligned address" pada nn.DataParallel (satu
    # tensor tied per scope). Lihat _detect_afpm_bank_size(). Ini yang
    # sebelumnya menyebabkan RuntimeError "size mismatch ... torch.Size([24])
    # ... torch.Size([1])" -- model lama selalu membuat Parameter(1,), padahal
    # checkpoint baru butuh Parameter(24,) / Parameter(16,) yang di-tie.
    decoder_bank_size = _detect_afpm_bank_size(state_dict, "decoder.") if decoder_act == "afpm" else None
    encoder_bank_size = _detect_afpm_bank_size(state_dict, "encoder.") if encoder_act == "afpm" else None

    _patch_smp_decoder_activation(model, decoder_act, ckpt_name, bank_size=decoder_bank_size)
    _patch_smp_encoder_activation(model, encoder_act, ckpt_name, bank_size=encoder_bank_size)

    model.load_state_dict(state_dict, strict=True)
    model._smp = True
    model._smp_classes = num_classes
    model._in_channels = in_channels
    model._decoder_activation = decoder_act
    model._encoder_activation = encoder_act
    model._input_h = int(input_h) if input_h is not None else None
    model._input_w = int(input_w) if input_w is not None else None
    model._input_scale = float(scale) if scale is not None else None
    model._ckpt_cfg = ckpt_cfg
    model.to(device).eval()
    logger.info(
        f"SMP U-Net dimuat: '{ckpt_name}' | encoder={encoder_name} (act={encoder_act}) | "
        f"decoder act={decoder_act} | in_channels={in_channels} | classes={num_classes} | "
        f"input_size={model._input_h}x{model._input_w} | device={device}"
    )
    if model._input_h is None or model._input_w is None:
        logger.warning(
            f"SMP U-Net '{ckpt_name}': checkpoint config tidak menyimpan input_h/input_w -- "
            f"inferensi akan memakai resolusi citra asli tanpa resize. Kalau checkpoint ini "
            f"dilatih pada resolusi fixed tertentu, hasil bisa tidak konsisten dengan training."
        )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# BUILDER: Vanilla U-Net kustom
# ══════════════════════════════════════════════════════════════════════════════

def _build_vanilla_model(
    state_dict: dict,
    ckpt_name: str,
    device: torch.device,
    ckpt_cfg: dict | None = None,
    meta: dict | None = None,
) -> UNet:
    ckpt_cfg = ckpt_cfg or {}
    meta = meta or {}

    act_type = _detect_vanilla_activation(state_dict, ckpt_name, ckpt_cfg)
    in_channels = _detect_in_channels(state_dict, ckpt_cfg, meta)
    base_features = int(ckpt_cfg.get("base_features", meta.get("base_features", 64)))
    scale = float(ckpt_cfg.get("scale", meta.get("scale", meta.get("input_scale", 0.5))))
    mask_values = _normalize_mask_values(meta.get("mask_values", ckpt_cfg.get("mask_values", [0, 1])))

    model = UNet(in_channels=in_channels, base_features=base_features, act_name=act_type)
    adapted_state_dict = _adapt_state_dict_shapes(state_dict, model)
    missing, unexpected = model.load_state_dict(adapted_state_dict, strict=False)

    if missing:
        logger.warning(f"Vanilla U-Net '{ckpt_name}': {len(missing)} key tidak cocok. Contoh: {missing[:3]}")
    if unexpected:
        logger.warning(f"Vanilla U-Net '{ckpt_name}': {len(unexpected)} key unexpected. Contoh: {unexpected[:3]}")

    model._smp = False
    model._ckpt_cfg = ckpt_cfg
    model._mask_values = mask_values
    model._input_scale = scale
    model._in_channels = in_channels
    model._num_classes = 2
    model._activation = act_type
    model._preprocess_image_like_local = preprocess_image_like_local
    model._preprocess_mask_like_local = preprocess_mask_like_local

    model.to(device).eval()
    logger.info(
        f"Vanilla U-Net dimuat: '{ckpt_name}' | aktivasi={act_type.upper()} | "
        f"in_channels={in_channels} | base_features={base_features} | scale={scale} | device={device}"
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# DISPATCHER UTAMA — load_unet()
# ══════════════════════════════════════════════════════════════════════════════

def load_unet(weights_path: str):
    if weights_path in _unet_cache:
        return _unet_cache[weights_path]

    p = Path(weights_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint U-Net tidak ditemukan: '{p}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state_dict, ckpt_cfg, meta = _load_checkpoint_and_extract(p, device)
    except Exception as e:
        raise RuntimeError(f"Gagal muat U-Net dari '{p}': {e}") from e

    if _is_smp_checkpoint(state_dict):
        logger.info(f"Terdeteksi SMP checkpoint: '{p.name}'")
        model = _build_smp_model(state_dict, ckpt_cfg, p.name, device)
    else:
        logger.info(f"Terdeteksi vanilla U-Net checkpoint: '{p.name}'")
        model = _build_vanilla_model(state_dict, p.name, device, ckpt_cfg=ckpt_cfg, meta=meta)

    _unet_cache[weights_path] = model
    return model


def clear_cache() -> None:
    _unet_cache.clear()
