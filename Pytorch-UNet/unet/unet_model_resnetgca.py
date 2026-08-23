"""
U-Net + GCA: Standard U-Net with Grouped Coordinate Attention
Paper: Ding & Gao (2025). GCA-ResUNet: Medical Image Segmentation Using
       Grouped Coordinate Attention. arXiv:2512.23990v1 [cs.CV] 30 Dec 2025.

ONLY MODIFICATION vs standard U-Net:
    GroupedCoordinateAttention (GCA) is appended after each encoder
    DoubleConv block. The decoder and all other components are UNCHANGED.

GCA placement rationale (Paper Section 3.1, Page 8):
    "GCA is inserted after the third convolution and batch normalization layer
     and before residual summation within each ResNet50 bottleneck."
    For standard U-Net (no residual), the equivalent placement is:
    after each encoder DoubleConv block and before the skip connection /
    MaxPool downsampling -- exactly preserving the paper's philosophy of
    applying GCA to semantically rich features before they are passed on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# GroupedCoordinateAttention (GCA)
# SOURCE: Ding & Gao (2025) Section 3.3, Equations (2)-(7), Figure 2.
#
# Eq.(2): f_h_avg = AvgPool_h(Xg),  f_h_max = MaxPool_h(Xg)  -> [B,Cg,H,1]
# Eq.(3): f_w_avg = AvgPool_w(Xg),  f_w_max = MaxPool_w(Xg)  -> [B,Cg,1,W]
# Eq.(4): A = sigmoid(Conv1x1(delta(BN(Conv1x1(F)))))
#         F = fused pooled features; delta=ReLU, sigma=Sigmoid
# Eq.(5): A_h in [B,Cg,H,1],  A_w in [B,Cg,1,W]  (directional split)
# Eq.(6): Yg = Xg (x) A_h (x) A_w   ((x) = element-wise multiply)
# Eq.(7): Y  = Concat(Y1, Y2, ..., YG)
#
# Default hyperparameters from ablation studies:
#   groups=2    (Table 5, Page 20): best DSC=86.11%, optimal channel grouping
#   reduction=2 (Table 6, Page 20): best DSC=86.11%, HD95=18.96
#   avg+max     (Table 7, Page 21): best over avg-only (82.04) or max-only (79.27)
# ===========================================================================

class GroupedCoordinateAttention(nn.Module):
    """
    GCA Module -- Ding & Gao (2025) Section 3.3.

    Args:
        in_channels : total input channels C
        groups      : number of channel groups G (default=2, Table 5 Page 20)
        reduction   : channel reduction ratio r  (default=2, Table 6 Page 20)
    """
    def __init__(self, in_channels, groups=2, reduction=2):
        super().__init__()

        assert in_channels % groups == 0, (
            f"in_channels ({in_channels}) must be divisible by groups ({groups})")

        self.groups = groups
        self.Cg     = in_channels // groups                      # Cg = C/G (Page 11)
        self.Cr     = max(8, self.Cg // reduction)               # Cg/r, min=8

        # Shared bottleneck transformation per group (Eq.4, Page 12)
        # First  1x1 conv: Cg -> Cg/r  (channel reduction)
        # Second 1x1 conv: Cg/r -> Cg  (channel restoration)
        self.conv1 = nn.Conv2d(self.Cg, self.Cr, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.Cr)
        self.conv2 = nn.Conv2d(self.Cr, self.Cg, 1, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns: [B, C, H, W]  (same shape, attention-weighted)
        """
        B, C, H, W = x.size()

        # Partition into G groups along channel dim (Page 11)
        xg = x.view(B, self.groups, self.Cg, H, W)

        outs = []
        for g in range(self.groups):
            Xg = xg[:, g]                          # [B, Cg, H, W]

            # ── Eq.(2): Horizontal pooling (pool W dimension, keep H) ────────
            fh_avg = F.adaptive_avg_pool2d(Xg, (H, 1))   # [B, Cg, H, 1]
            fh_max = F.adaptive_max_pool2d(Xg, (H, 1))   # [B, Cg, H, 1]

            # ── Eq.(3): Vertical pooling (pool H dimension, keep W) ──────────
            fw_avg = F.adaptive_avg_pool2d(Xg, (1, W))   # [B, Cg, 1, W]
            fw_max = F.adaptive_max_pool2d(Xg, (1, W))   # [B, Cg, 1, W]

            # Combine avg+max for both directions (Table 7: avg+max is best)
            fh = fh_avg + fh_max                          # [B, Cg, H, 1]
            fw = fw_avg + fw_max                          # [B, Cg, 1, W]

            # Fuse H and W via broadcast addition (Figure 2, summation-based fusion)
            # [B,Cg,H,1] + [B,Cg,1,W] broadcasts to [B,Cg,H,W]
            F_fused = fh + fw                             # [B, Cg, H, W]

            # ── Eq.(4): Attention generation ─────────────────────────────────
            # A = sigma( Conv1x1( delta( BN( Conv1x1(F) ) ) ) )
            A = self.conv1(F_fused)       # [B, Cr, H, W]  (channel reduction)
            A = F.relu(self.bn1(A))       # delta(BN(.))
            A = self.conv2(A)             # [B, Cg, H, W]  (channel restore)
            A = torch.sigmoid(A)          # sigma(.)

            # ── Eq.(5): Split A into directional components ───────────────────
            Ah = F.adaptive_avg_pool2d(A, (H, 1))  # [B, Cg, H, 1]
            Aw = F.adaptive_avg_pool2d(A, (1, W))  # [B, Cg, 1, W]

            # ── Eq.(6): Group refinement Yg = Xg (x) Ah (x) Aw ──────────────
            # Broadcasting: [B,Cg,H,W] * [B,Cg,H,1] * [B,Cg,1,W]
            Yg = Xg * Ah * Aw             # [B, Cg, H, W]
            outs.append(Yg)

        # ── Eq.(7): Y = Concat(Y1, ..., YG) ──────────────────────────────────
        return torch.cat(outs, dim=1)     # [B, C, H, W]


# ===========================================================================
# DoubleConv -- standard U-Net block (UNCHANGED from baseline)
# SOURCE: Ronneberger et al. (2015) U-Net, preserved as-is.
# ===========================================================================

class DoubleConv(nn.Module):
    """Two consecutive 3x3 Conv + BN + ReLU blocks (standard U-Net)."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


# ===========================================================================
# DoubleConvWithGCA
# THE ONLY ARCHITECTURAL MODIFICATION vs standard U-Net.
#
# GCA is appended immediately after DoubleConv in the encoder stages.
# This mirrors the paper's strategy: GCA follows the last conv+BN of each
# bottleneck (Section 3.1, Page 8) to modulate semantically rich features
# before they feed downstream (skip connection or MaxPool downsampling).
#
# All encoder output channels in standard U-Net (64,128,256,512,1024) are
# divisible by groups=2, so GCA applies without any channel constraint issue.
# ===========================================================================

class DoubleConvWithGCA(nn.Module):
    """
    Standard DoubleConv followed by GCA.
    Used in the ENCODER only; decoder uses plain DoubleConv (paper-faithful).

    Paper: GCA is a plug-and-play module applied to the encoder backbone
           (Section 3.3, Page 11). Decoder is kept standard (Section 3.4).
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 gca_groups=2, gca_reduction=2):
        super().__init__()
        self.dconv = DoubleConv(in_channels, out_channels, mid_channels)
        # GCA: groups=2 (Table 5), reduction=2 (Table 6)
        self.gca   = GroupedCoordinateAttention(out_channels, gca_groups, gca_reduction)

    def forward(self, x):
        x = self.dconv(x)   # standard DoubleConv
        x = self.gca(x)     # GCA attention -- the only modification
        return x


# ===========================================================================
# Down -- MaxPool + DoubleConvWithGCA  (encoder downsampling)
# MaxPool: standard, unchanged.
# DoubleConvWithGCA: GCA added after DoubleConv (the only change).
# ===========================================================================

class Down(nn.Module):
    """Encoder down-step: MaxPool2d(2) followed by DoubleConvWithGCA."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWithGCA(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


# ===========================================================================
# Up -- UNCHANGED from standard U-Net (decoder, paper Section 3.4)
# Paper (Page 14): "Bilinear interpolation ... concatenated with encoder
#                   feature ... two successive 3x3 convolutional layers"
# GCA is NOT added to the decoder -- only the encoder uses GCA.
# ===========================================================================

class Up(nn.Module):
    """Decoder up-step: upsample -> concat skip -> DoubleConv (NO GCA)."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dY = x2.size(2) - x1.size(2)
        dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ===========================================================================
# OutConv -- UNCHANGED from standard U-Net
# ===========================================================================

class OutConv(nn.Module):
    """1x1 conv output head -- unchanged from standard U-Net."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


# ===========================================================================
# UNet + GCA
# Encoder: inc + down1-down4 each have GCA appended after DoubleConv.
# Decoder: up1-up4 are standard U-Net (no GCA). OutConv unchanged.
# ===========================================================================

class UNet(nn.Module):
    """
    Standard U-Net with Grouped Coordinate Attention (GCA) in the encoder.

    MODIFICATION SUMMARY (vs standard U-Net):
        ONLY the encoder DoubleConv blocks are wrapped with GCA.
        Decoder (Up blocks), output head, skip connections, MaxPool,
        and all training settings are IDENTICAL to baseline U-Net.

    WHY ENCODER ONLY:
        The paper (Section 3.4, Page 13) explicitly keeps the decoder
        standard: bilinear upsample -> concat -> two 3x3 convs with ReLU.
        GCA is an encoder-side module in the paper; adding it to the decoder
        would deviate from the paper's design and ablation scope.

    CHANNEL COMPATIBILITY:
        Encoder output channels: 64, 128, 256, 512, 1024.
        All divisible by groups=2. GCA applies cleanly at every encoder stage.

    GCA DEFAULT SETTINGS (from paper ablation, Tables 5-7):
        groups=2    -- optimal channel grouping  (Table 5, Page 20)
        reduction=2 -- optimal bottleneck ratio  (Table 6, Page 20)
        avg+max     -- complementary statistics  (Table 7, Page 21)
    """
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear
        factor = 2 if bilinear else 1

        # ── Encoder: DoubleConv + GCA at every stage ──────────────────────────
        # inc: 3 -> 64 channels,  GCA(64, groups=2, reduction=2)
        self.inc   = DoubleConvWithGCA(n_channels, 64)

        # down1: 64 -> 128,  GCA(128, groups=2, reduction=2)
        self.down1 = Down(64,  128)

        # down2: 128 -> 256,  GCA(256, groups=2, reduction=2)
        self.down2 = Down(128, 256)

        # down3: 256 -> 512,  GCA(512, groups=2, reduction=2)
        self.down3 = Down(256, 512)

        # down4: 512 -> 1024,  GCA(1024, groups=2, reduction=2)
        self.down4 = Down(512, 1024 // factor)

        # ── Decoder: UNCHANGED standard U-Net (no GCA) ────────────────────────
        self.up1   = Up(1024, 512  // factor, bilinear)
        self.up2   = Up(512,  256  // factor, bilinear)
        self.up3   = Up(256,  128  // factor, bilinear)
        self.up4   = Up(128,  64,             bilinear)

        # ── Output head: UNCHANGED ─────────────────────────────────────────────
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder (with GCA at each stage)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder (standard, no GCA)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        return self.outc(x)
