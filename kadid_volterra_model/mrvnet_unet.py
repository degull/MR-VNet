# mrvnet_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from kadid_volterra_model.mrvnet_block import MRVNetBlock

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, shift_radius=1, rank=4):
        super(MRVNetUNet, self).__init__()

        self.enc1 = nn.Sequential(*[MRVNetBlock(in_channels if i == 0 else base_channels, base_channels, shift_radius, rank) for i in range(4)])
        self.enc2 = nn.Sequential(*[MRVNetBlock(base_channels if i == 0 else base_channels * 2, base_channels * 2, shift_radius, rank) for i in range(4)])
        self.enc3 = nn.Sequential(*[MRVNetBlock(base_channels * 2 if i == 0 else base_channels * 4, base_channels * 4, shift_radius, rank) for i in range(4)])
        self.enc4 = nn.Sequential(*[MRVNetBlock(base_channels * 4 if i == 0 else base_channels * 8, base_channels * 8, shift_radius, rank) for i in range(4)])

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, shift_radius, rank)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, shift_radius, rank)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, shift_radius, rank)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, shift_radius, rank)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.down(x1))
        x3 = self.enc3(self.down(x2))
        x4 = self.enc4(self.down(x3))
        m = self.middle(self.down(x4))

        d3 = self.dec3(torch.cat([self.up3(m), x4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x2], dim=1))

        d1_upsampled = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_conv(d1_upsampled + x1)

        return out
