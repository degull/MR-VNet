# mrvnet_unet
import torch
import torch.nn as nn
import torch.nn.functional as F
from mr_vnet_model.mrvnet_block import MRVNetBlock

class MRVNetUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, rank=4):
        super().__init__()
        self.enc1 = MRVNetBlock(in_channels, base_channels, num_layers=4)
        self.enc2 = MRVNetBlock(base_channels, base_channels * 2, num_layers=4)
        self.enc3 = MRVNetBlock(base_channels * 2, base_channels * 4, num_layers=4)
        self.enc4 = MRVNetBlock(base_channels * 4, base_channels * 8, num_layers=4)
        self.down = nn.MaxPool2d(2)
        self.middle = MRVNetBlock(base_channels * 8, base_channels * 8, num_layers=1)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.dec3 = MRVNetBlock(base_channels * 4 + base_channels * 8, base_channels * 4, num_layers=1, rank=rank)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.dec2 = MRVNetBlock(base_channels * 2 + base_channels * 4, base_channels * 2, num_layers=1, rank=rank)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.dec1 = MRVNetBlock(base_channels + base_channels * 2, base_channels, num_layers=1, rank=rank)
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
        d1_upsampled = F.interpolate(d1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        out = self.final(d1_upsampled + x1)
        return out
