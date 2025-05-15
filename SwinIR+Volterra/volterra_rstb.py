# volterra_rstb.py
# volterra_rstb.py
from swinir import SwinTransformerBlock, PatchEmbed, PatchUnEmbed
from volterra_layer import VolterraLayer2D
import torch.nn as nn

class RSTBWithVolterra(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=224, patch_size=4, resi_connection='1conv',
                 rank=4):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.patch_unembed = PatchUnEmbed(img_size, patch_size, in_chans=0, embed_dim=dim)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=0, embed_dim=dim)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ))
            self.blocks.append(VolterraLayer2D(dim, dim, rank=rank))  # ✅ Volterra 삽입

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1)
            )

    def forward(self, x, x_size):
        res = x
        for layer in self.blocks:
            if isinstance(layer, SwinTransformerBlock):
                x = layer(x, x_size)
            else:
                x = self.patch_unembed(x, x_size)
                x = layer(x)
                x = self.patch_embed(x)
        x = self.patch_unembed(x, x_size)
        x = self.conv(x)
        x = self.patch_embed(x)
        return x + res
