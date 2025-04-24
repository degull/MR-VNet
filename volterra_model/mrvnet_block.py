# mrvnet_block.py
import torch.nn as nn
from mr_vnet_model.volterra_layer import VolterraLayer2D

class MRVNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1, rank=4):
        super().__init__()
        self.block = nn.Sequential(*[
            VolterraLayer2D(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, rank=rank)
            for i in range(num_layers)
        ])

    def forward(self, x):
        return self.block(x)
