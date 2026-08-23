"""
ResNet50-UNet  —  ResNet50 encoder (pretrained ImageNet) + U-Net decoder.

Architecture summary
--------------------
Encoder  (frozen or fine-tuned ResNet50 stages):
  enc0: conv1 + bn1 + relu          ->  64 ch,  H/2  x W/2
  enc1: maxpool + layer1 (3 blocks) -> 256 ch,  H/4  x W/4
  enc2: layer2 (4 blocks)           -> 512 ch,  H/8  x W/8
  enc3: layer3 (6 blocks)           -> 1024ch,  H/16 x W/16
  enc4: layer4 (3 blocks)           -> 2048ch,  H/32 x W/32  [bottleneck]

Decoder  (randomly initialised, trained from scratch):
  dec4: up(2048) + skip(1024)       ->  512ch,  H/16
  dec3: up(512)  + skip(512)        ->  256ch,  H/8
  dec2: up(256)  + skip(256)        ->  128ch,  H/4
  dec1: up(128)  + skip(64)         ->   64ch,  H/2
  dec0: up(64),  no skip            ->   32ch,  H    [full resolution]
  outc: Conv1x1(32, n_classes)

Skip connections reuse the encoder feature maps at matching spatial scales,
following the standard U-Net design (Ronneberger et al. 2015).

Each DecoderBlock:  ConvTranspose2d  ->  cat(encoder_skip)  ->  DoubleConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
    from torchvision.models import ResNet50_Weights
    _NEW_TV = True
except (ImportError, AttributeError):
    from torchvision import models
    _NEW_TV = False


# ── DoubleConv ────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU  (standard U-Net block)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv(x)


# ── DecoderBlock ──────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    """
    Decoder stage: ConvTranspose2d -> concat(skip) -> DoubleConv.

    in_channels  : channels arriving from the previous (deeper) decoder stage
    skip_channels: channels of the matching encoder skip connection
    out_channels : desired output channels after DoubleConv

    Channel flow:
      ConvTranspose2d(in_channels, in_channels//2)    [upsample + halve ch]
      cat( upsampled[in_channels//2] , skip[skip_channels] )
      DoubleConv(in_channels//2 + skip_channels, out_channels)
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                       kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x  = self.up(x)
        # pad to match skip spatial size (handles odd dimensions)
        dY = skip.size(2) - x.size(2)
        dX = skip.size(3) - x.size(3)
        if dY > 0 or dX > 0:
            x = F.pad(x, [dX // 2, dX - dX // 2,
                          dY // 2, dY - dY // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# ── ResNet50-UNet ─────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    ResNet50-UNet: pretrained ResNet50 encoder + randomly-initialised U-Net decoder.

    Args:
        n_channels (int): input image channels (3 for RGB; adapted automatically
                          via a learned 1x1 conv if n_channels != 3).
        n_classes  (int): number of segmentation output classes.
        bilinear   (bool): unused; kept for drop-in compatibility with train.py
                           that calls UNet(n_channels, n_classes, bilinear=False).
    """
    def __init__(self, n_channels=3, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes

        # ── Input adapter  (only when n_channels != 3) ────────────────────────
        self.input_adapter = (nn.Conv2d(n_channels, 3, 1, bias=False)
                              if n_channels != 3 else nn.Identity())

        # ── Encoder: ResNet50 pretrained on ImageNet ───────────────────────────
        if _NEW_TV:
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = models.resnet50(pretrained=True)

        # stage 0 — stem: 64ch, H/2 x W/2
        self.enc0 = nn.Sequential(backbone.conv1,
                                  backbone.bn1,
                                  backbone.relu)
        # stage 1 — maxpool + layer1: 256ch, H/4 x W/4
        self.enc1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        # stage 2 — layer2: 512ch, H/8 x W/8
        self.enc2 = backbone.layer2
        # stage 3 — layer3: 1024ch, H/16 x W/16
        self.enc3 = backbone.layer3
        # stage 4 — layer4 (bottleneck): 2048ch, H/32 x W/32
        self.enc4 = backbone.layer4

        # ── Decoder: U-Net style, ConvTranspose2d upsampling ──────────────────
        # dec4: 2048 -> 512  (skip from enc3: 1024ch)
        #   ConvTranspose(2048->1024) + cat(1024) = 2048 -> DoubleConv -> 512
        self.dec4 = DecoderBlock(2048, 1024, 512)

        # dec3: 512 -> 256  (skip from enc2: 512ch)
        #   ConvTranspose(512->256) + cat(512) = 768 -> DoubleConv -> 256
        self.dec3 = DecoderBlock(512,  512,  256)

        # dec2: 256 -> 128  (skip from enc1: 256ch)
        #   ConvTranspose(256->128) + cat(256) = 384 -> DoubleConv -> 128
        self.dec2 = DecoderBlock(256,  256,  128)

        # dec1: 128 -> 64   (skip from enc0: 64ch)
        #   ConvTranspose(128->64) + cat(64) = 128 -> DoubleConv -> 64
        self.dec1 = DecoderBlock(128,  64,   64)

        # dec0: 64 -> 32, no skip, restores full H x W
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(32, 32))

        # ── Output head ───────────────────────────────────────────────────────
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Input adapter
        x  = self.input_adapter(x)

        # Encoder
        e0 = self.enc0(x)    #  64ch,  H/2
        e1 = self.enc1(e0)   # 256ch,  H/4
        e2 = self.enc2(e1)   # 512ch,  H/8
        e3 = self.enc3(e2)   # 1024ch, H/16
        e4 = self.enc4(e3)   # 2048ch, H/32  (bottleneck)

        # Decoder with skip connections
        d4 = self.dec4(e4, e3)   # 512ch,  H/16
        d3 = self.dec3(d4, e2)   # 256ch,  H/8
        d2 = self.dec2(d3, e1)   # 128ch,  H/4
        d1 = self.dec1(d2, e0)   #  64ch,  H/2
        d0 = self.dec0(d1)       #  32ch,  H x W

        return self.outc(d0)     # n_classes, H x W
