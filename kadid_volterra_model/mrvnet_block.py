import torch
import torch.nn as nn
from kadid_volterra_model.volterra_layer import VolterraLayer2D

class MRVNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shift_radius=1, rank=4):
        super(MRVNetBlock, self).__init__()
        # 하나의 Volterra Layer만 (Activation 없음)
        self.block = VolterraLayer2D(
            in_channels=in_channels,
            out_channels=out_channels,
            shift_radius=shift_radius,
            rank=rank
        )

    def forward(self, x):
        return self.block(x)
