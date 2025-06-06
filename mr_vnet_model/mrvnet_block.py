
# mrvnet_block.py
# VolterraLayer들을 쌓은 block (U-Net의 building block)
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