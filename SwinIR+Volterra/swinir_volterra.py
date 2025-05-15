# swinir_volterra.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swinir import SwinIR as SwinIRBase
from volterra_layer import VolterraLayer2D
from volterra_rstb import RSTBWithVolterra
import torch.nn as nn
import torch.nn.functional as F

class SwinIRVolterra(SwinIRBase):
    def __init__(self, *args, rank=4, **kwargs):
        super().__init__(*args, **kwargs)

        # ① Shallow Conv 뒤
        self.volterra_shallow = VolterraLayer2D(self.embed_dim, self.embed_dim, rank=rank)

        # ② RSTB with Volterra block마다 삽입된 구조로 대체
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTBWithVolterra(
                dim=self.embed_dim,
                input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer+1])],
                norm_layer=self.norm_layer,
                use_checkpoint=self.use_checkpoint,
                img_size=self.img_size,
                patch_size=self.patch_size,
                resi_connection=self.resi_connection,
                rank=rank
            )
            self.layers.append(layer)

        # ③ Upsample 이후
        self.volterra_upsample = VolterraLayer2D(self.embed_dim, self.embed_dim, rank=rank)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # ① Shallow Conv + Volterra
        x = self.conv_first(x)
        x = self.volterra_shallow(x)

        # ② Swin Transformer Block + Volterra 반복 구조
        x_feat = self.forward_features(x)
        x = self.conv_after_body(x_feat) + x

        # Upsample
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
        elif self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_hr(x))

        # ③ 마지막 Volterra + Output
        x = self.volterra_upsample(x)
        x = self.conv_last(x)
        x = x / self.img_range + self.mean
        return x[:, :, :H * self.upscale, :W * self.upscale]
