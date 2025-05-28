# swin_transformer_volterra.py
# (SwinIR + VolterraLayer)

# swin_transformer_volterra.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from mr_vnet_model.volterra_layer import VolterraLayer2D


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class SwinTransformerLayerVolterra(nn.Module):
    def __init__(self, dim, num_heads, rank=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)
        self.volterra = VolterraLayer2D(dim, dim, kernel_size=3, rank=rank)

    def forward(self, x, H, W):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + shortcut

        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = x + self.volterra(x)
        x = x.view(B, C, -1).transpose(1, 2)

        x = x + self.mlp(self.norm2(x))
        return x


class RSTBBlockVolterra(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, depth=6, rank=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.layers = nn.ModuleList([
            SwinTransformerLayerVolterra(dim, num_heads, rank=rank) for _ in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        for layer in self.layers:
            x_flat = layer(x_flat, H, W)

        x_feat = x_flat.transpose(1, 2).view(B, C, H, W)
        x_out = self.conv(x_feat)
        return x + x_out
