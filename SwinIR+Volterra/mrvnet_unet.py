# mrvnet_unet.py
# Middle Block에 Swin + Volterra Hybrid 적용 가능

# mrvnet_unet.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from mr_vnet_model.mrvnet_block import MRVNetBlock
from swin_transformer_volterra import RSTBBlockVolterra

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, rank=4, use_lossless=False, use_swin_middle=True):
        super().__init__()

        # Encoder
        self.enc1 = MRVNetBlock(in_channels, base_channels, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc2 = MRVNetBlock(base_channels, base_channels * 2, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc3 = MRVNetBlock(base_channels * 2, base_channels * 4, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc4 = MRVNetBlock(base_channels * 4, base_channels * 8, num_layers=4, rank=rank, use_lossless=use_lossless)

        self.down = nn.MaxPool2d(2)

        # swin 추가버전
        # ✅ Middle Block: SwinIR-style RSTB with Volterra
        if use_swin_middle:
            self.middle = RSTBBlockVolterra(
                dim=base_channels * 8,
                input_resolution=(16, 16),
                num_heads=8,
                depth=6,
                rank=rank
            )
        else:
            self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, num_layers=1, rank=rank, use_lossless=use_lossless)



        # swin 사용x 버전
        model = MRVNetUNet(
            in_channels=3,
            base_channels=32,
            rank=4,
            use_lossless=True,
            use_swin_middle=False  # ✅ Swin 제거
        ).to()

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up0 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec0 = MRVNetBlock(base_channels + base_channels, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)


        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(self.down(x3))

        m = self.middle(self.down(x4))

        d3 = self.dec3(torch.cat([self.up3(m), x4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), x1], dim=1))

        return self.final(d0)
