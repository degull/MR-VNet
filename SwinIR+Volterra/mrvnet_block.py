# mrvnet_block.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mr_vnet_model.volterra_layer import VolterraLayer2D
import torch.nn as nn

class MRVNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, rank=4, use_lossless=False):
        super().__init__()
        layers = []
        for i in range(num_layers):
            inc = in_channels if i == 0 else out_channels
            layers.append(VolterraLayer2D(inc, out_channels, kernel_size=3, rank=rank, use_lossless=use_lossless))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
