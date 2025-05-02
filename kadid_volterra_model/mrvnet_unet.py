import torch
import torch.nn as nn
import torch.nn.functional as F
from kadid_volterra_model.mrvnet_block import MRVNetBlock

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, shift_radius=1, rank=4):
        super(MRVNetUNet, self).__init__()

        # ✅ Encoder (4 blocks × 4 Volterra Layers = 16)
        self.enc1 = nn.Sequential(
            MRVNetBlock(in_channels, base_channels, shift_radius, rank),
            MRVNetBlock(base_channels, base_channels, shift_radius, rank),
            MRVNetBlock(base_channels, base_channels, shift_radius, rank),
            MRVNetBlock(base_channels, base_channels, shift_radius, rank)
        )

        self.enc2 = nn.Sequential(
            MRVNetBlock(base_channels, base_channels * 2, shift_radius, rank),
            MRVNetBlock(base_channels * 2, base_channels * 2, shift_radius, rank),
            MRVNetBlock(base_channels * 2, base_channels * 2, shift_radius, rank),
            MRVNetBlock(base_channels * 2, base_channels * 2, shift_radius, rank)
        )

        self.enc3 = nn.Sequential(
            MRVNetBlock(base_channels * 2, base_channels * 4, shift_radius, rank),
            MRVNetBlock(base_channels * 4, base_channels * 4, shift_radius, rank),
            MRVNetBlock(base_channels * 4, base_channels * 4, shift_radius, rank),
            MRVNetBlock(base_channels * 4, base_channels * 4, shift_radius, rank)
        )

        self.enc4 = nn.Sequential(
            MRVNetBlock(base_channels * 4, base_channels * 8, shift_radius, rank),
            MRVNetBlock(base_channels * 8, base_channels * 8, shift_radius, rank),
            MRVNetBlock(base_channels * 8, base_channels * 8, shift_radius, rank),
            MRVNetBlock(base_channels * 8, base_channels * 8, shift_radius, rank)
        )

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        # ✅ Middle Block (1 Volterra Layer)
        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, shift_radius, rank)

        # ✅ Decoder (4 Volterra Layers)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MRVNetBlock(base_channels * 12, base_channels * 4, shift_radius, rank)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MRVNetBlock(base_channels * 6, base_channels * 2, shift_radius, rank)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MRVNetBlock(base_channels * 3, base_channels, shift_radius, rank)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # ✅ Encoder
        x1 = self.enc1(x)              # [B, base, H, W]
        x2 = self.enc2(self.down(x1))  # [B, base*2, H/2, W/2]
        x3 = self.enc3(self.down(x2))  # [B, base*4, H/4, W/4]
        x4 = self.enc4(self.down(x3))  # [B, base*8, H/8, W/8]

        # ✅ Middle Block
        m = self.middle(self.down(x4)) # [B, base*8, H/16, W/16]

        # ✅ Decoder
        d3 = self.up3(m)                        # [B, base*4, H/8, W/8]
        d3 = torch.cat([d3, x4], dim=1)         # [B, base*12, H/8, W/8]
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                       # [B, base*2, H/4, W/4]
        d2 = torch.cat([d2, x3], dim=1)         # [B, base*6, H/4, W/4]
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                       # [B, base, H/2, W/2]
        d1 = torch.cat([d1, x2], dim=1)         # [B, base*3, H/2, W/2]
        d1 = self.dec1(d1)

        # ✅ 최종 출력
        d1_upsampled = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_conv(d1_upsampled + x1)

        return out
