""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================
# 1. GCA MODULE (Sama seperti sebelumnya)
# ========================================================
class GroupedCoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, groups: int = 2, reduction: int = 2):
        super(GroupedCoordinateAttention, self).__init__()
        assert in_channels % groups == 0
        self.groups = groups
        self.channels_per_group = in_channels // groups
        self.reduced_channels = max(8, self.channels_per_group // reduction)

        self.conv1 = nn.Conv2d(self.channels_per_group, self.reduced_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(self.reduced_channels)
        self.conv2 = nn.Conv2d(self.reduced_channels, self.channels_per_group, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        x_groups = x.view(B, self.groups, self.channels_per_group, H, W)
        group_outputs = []
        for g in range(self.groups):
            xg = x_groups[:, g]
            f_h_avg = F.adaptive_avg_pool2d(xg, (H, 1))
            f_h_max = F.adaptive_max_pool2d(xg, (H, 1))
            f_w_avg = F.adaptive_avg_pool2d(xg, (1, W))
            f_w_max = F.adaptive_max_pool2d(xg, (1, W))
            f_h = f_h_avg + f_h_max
            f_w = f_w_avg + f_w_max
            f_combined = f_h + f_w
            f_att = self.conv1(f_combined)
            f_att = F.relu(self.bn1(f_att))
            f_att = torch.sigmoid(self.conv2(f_att))
            a_h = F.adaptive_avg_pool2d(f_att, (H, 1))
            a_w = F.adaptive_avg_pool2d(f_att, (1, W))
            yg = xg * a_h * a_w
            group_outputs.append(yg)
        return torch.cat(group_outputs, dim=1)

# ========================================================
# 2. U-NET BLOCKS (Mengikuti pola penamaan Milesial)
# ========================================================
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConvGCA(nn.Module):
    """Double Conv dilanjutkan dengan GCA.
       Sesuai dengan struktur state_dict: 'dconv' lalu 'gca' """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dconv = DoubleConv(in_channels, out_channels)
        self.gca = GroupedCoordinateAttention(out_channels)

    def forward(self, x):
        x = self.dconv(x)
        x = self.gca(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv + GCA"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvGCA(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ========================================================
# 3. KELAS UNET UTAMA (Milesial GCA Version)
# ========================================================
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # inc.dconv & inc.gca
        self.inc = DoubleConvGCA(n_channels, 64) 
        
        # down1 sampai down4
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder standar milesial (TANPA GCA)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits