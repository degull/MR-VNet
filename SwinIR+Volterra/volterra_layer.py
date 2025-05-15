# volterra_layer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, shift_radius=1, rank=4):
        super().__init__()
        self.linear_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.rank = rank
        self.shift_offsets = [(dy, dx) for dy in range(-shift_radius, shift_radius + 1)
                                         for dx in range(-shift_radius, shift_radius + 1)]
        self.W2a = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(rank)])
        self.W2b = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 1) for _ in range(rank)])

    def circular_shift(self, x, shift):
        return torch.roll(x, shifts=shift, dims=(2, 3))

    def forward(self, x):
        F1 = self.linear_conv(x)
        shifted_maps = [self.circular_shift(x, shift) for shift in self.shift_offsets]
        F2 = 0
        count = 0
        for i, Xi in enumerate(shifted_maps):
            for j, Xj in enumerate(shifted_maps):
                if j < i:
                    continue
                prod = Xi * Xj
                for q in range(self.rank):
                    F2 += self.W2b[q](self.W2a[q](prod))
                count += 1
        F2 = F2 / count
        return F1 + F2
