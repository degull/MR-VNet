# mrvnet_unet.py
#  U-Net 복원 구조이며, 내부 모든 block은 Volterra 기반으로 구성되어 일반 CNN U-Net보다 더 정밀한 왜곡 복원 능력

""" import torch
import torch.nn as nn
import torch.nn.functional as F
from mr_vnet_model.mrvnet_block import MRVNetBlock

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, rank=4, use_lossless=False):
        super().__init__()

        # Encoder
        self.enc1 = MRVNetBlock(in_channels, base_channels, num_layers=4, rank=rank, use_lossless=use_lossless)             # out: 32
        self.enc2 = MRVNetBlock(base_channels, base_channels * 2, num_layers=4, rank=rank, use_lossless=use_lossless)       # out: 64
        self.enc3 = MRVNetBlock(base_channels * 2, base_channels * 4, num_layers=4, rank=rank, use_lossless=use_lossless)   # out: 128
        self.enc4 = MRVNetBlock(base_channels * 4, base_channels * 8, num_layers=4, rank=rank, use_lossless=use_lossless)   # out: 256

        self.down = nn.MaxPool2d(2)

        # Middle Block
        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, num_layers=1, rank=rank, use_lossless=use_lossless)  # 256 -> 256

        # Decoder (주의: skip connection으로 채널 수가 합쳐짐)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)                         # 256 -> 128
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, num_layers=1, rank=rank, use_lossless=use_lossless)  # 128+256 -> 128

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)                         # 128 -> 64
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, num_layers=1, rank=rank, use_lossless=use_lossless)  # 64+128 -> 64

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)                             # 64 -> 32
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)          # 32+64 -> 32

        self.up0 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)                                 # 32 -> 32
        self.dec0 = MRVNetBlock(base_channels + base_channels, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)              # 32+32 -> 32

        # Final output
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)           # [B, 32, H, W]
        x2 = self.enc2(self.down(x1))  # [B, 64, H/2, W/2]
        x3 = self.enc3(self.down(x2))  # [B, 128, H/4, W/4]
        x4 = self.enc4(self.down(x3))  # [B, 256, H/8, W/8]

        # Middle block
        m = self.middle(self.down(x4))  # [B, 256, H/16, W/16]

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(m), x4], dim=1))  # [B, 128, H/8, W/8]
        d2 = self.dec2(torch.cat([self.up2(d3), x3], dim=1)) # [B, 64, H/4, W/4]
        d1 = self.dec1(torch.cat([self.up1(d2), x2], dim=1)) # [B, 32, H/2, W/2]
        d0 = self.dec0(torch.cat([self.up0(d1), x1], dim=1)) # [B, 32, H, W]

        # Output
        out = self.final(d0)  # [B, 3, H, W]
        return out
 """

# derain 수정
import torch
import torch.nn as nn
import torch.nn.functional as F
from mr_vnet_model.mrvnet_block import MRVNetBlock


class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, rank=4, use_lossless=False):
        super().__init__()

        # Encoder
        self.enc1 = MRVNetBlock(in_channels, base_channels, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc2 = MRVNetBlock(base_channels, base_channels * 2, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc3 = MRVNetBlock(base_channels * 2, base_channels * 4, num_layers=4, rank=rank, use_lossless=use_lossless)
        self.enc4 = MRVNetBlock(base_channels * 4, base_channels * 8, num_layers=4, rank=rank, use_lossless=use_lossless)

        self.down = nn.MaxPool2d(2)

        # Middle Block
        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, num_layers=1, rank=rank, use_lossless=use_lossless)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)

        self.up0 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2)
        self.dec0 = MRVNetBlock(base_channels + base_channels, base_channels, num_layers=1, rank=rank, use_lossless=use_lossless)

        # Final output
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(self.down(x3))

        # Middle block
        m = self.middle(self.down(x4))

        # Decoder
        d3_up = self.up3(m)
        d3 = self.dec3(torch.cat([d3_up, x4], dim=1))

        d2_up = self.up2(d3)
        d2 = self.dec2(torch.cat([d2_up, x3], dim=1))

        d1_up = self.up1(d2)
        d1 = self.dec1(torch.cat([d1_up, x2], dim=1))

        d0_up = self.up0(d1)

        # ⚠️ 크기 mismatch 방지 (x1 크기에 맞춤)
        if d0_up.shape[-2:] != x1.shape[-2:]:
            d0_up = F.interpolate(d0_up, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        d0 = self.dec0(torch.cat([d0_up, x1], dim=1))

        out = self.final(d0)
        return out
