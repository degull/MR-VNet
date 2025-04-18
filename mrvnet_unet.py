# mrvnet_unet.py
#  U-Net 복원 구조이며, 내부 모든 block은 Volterra 기반으로 구성되어 일반 CNN U-Net보다 더 정밀한 왜곡 복원 능력
import torch
import torch.nn as nn
import torch.nn.functional as F
from mrvnet_block import MRVNetBlock

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, rank=4):
        super().__init__()

        # ✅ 인코더
        self.enc1 = MRVNetBlock(in_channels, base_channels, num_layers=4, rank=rank)
        self.enc2 = MRVNetBlock(base_channels, base_channels * 2, num_layers=4, rank=rank)
        self.enc3 = MRVNetBlock(base_channels * 2, base_channels * 4, num_layers=4, rank=rank)
        self.enc4 = MRVNetBlock(base_channels * 4, base_channels * 8, num_layers=4, rank=rank)

        self.down = nn.MaxPool2d(2)

        # ✅ Middle Block
        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, num_layers=1, rank=rank)

        # ✅ 디코더 + 업샘플링 + skip connection (concat 기준 채널 주의!)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, num_layers=1, rank=rank)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, num_layers=1, rank=rank)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, num_layers=1, rank=rank)

        # ✅ 최종 출력
        self.final = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 인코더
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(self.down(x3))

        # 중간 블록
        m = self.middle(self.down(x4))

        # 디코더
        d3 = self.dec3(torch.cat([self.up3(m), x4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x2], dim=1))

        # ⚠️ x1 (256x256)과 d1 (128x128)의 크기 mismatch → 업샘플링 필요
        d1_upsampled = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        # 최종 출력
        out = self.final(d1_upsampled + x1)  # skip connection with residual
        return out
