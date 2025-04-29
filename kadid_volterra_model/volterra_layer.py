# volterra_layer.py
import torch
import torch.nn as nn

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, shift_radius=1, rank=4):
        super(VolterraLayer2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.shift_radius = shift_radius
        self.rank = rank

        # ✅ 1차항 (선형 Convolution)
        self.linear_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # ✅ shift offsets: (-1,-1) ~ (1,1) 총 9개
        self.shift_offsets = [
            (dy, dx)
            for dy in range(-shift_radius, shift_radius + 1)
            for dx in range(-shift_radius, shift_radius + 1)
        ]

        # ✅ 각 (shift_i, shift_j) 조합마다 rank개씩 1x1 projection (W2a, W2b)
        self.projectors = nn.ModuleDict()
        for i, _ in enumerate(self.shift_offsets):
            for j, _ in enumerate(self.shift_offsets):
                self.projectors[f"{i}_{j}"] = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    ) for _ in range(rank)
                ])

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear_conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        for proj_list in self.projectors.values():
            for proj in proj_list:
                nn.init.kaiming_normal_(proj[0].weight, a=0, mode='fan_in', nonlinearity='linear')
                nn.init.kaiming_normal_(proj[1].weight, a=0, mode='fan_in', nonlinearity='linear')

    def circular_shift(self, x, shift):
        return torch.roll(x, shifts=shift, dims=(2, 3))

    def forward(self, x):
        # ✅ 1차항 (F1)
        F1 = self.linear_conv(x)

        # ✅ 2차항 (F2)
        shifted_maps = [self.circular_shift(x, shift) for shift in self.shift_offsets]
        F2 = 0
        for i, Xi in enumerate(shifted_maps):
            for j, Xj in enumerate(shifted_maps):
                pairwise_product = Xi * Xj
                for q in range(self.rank):
                    W2a = self.projectors[f"{i}_{j}"][q][0]
                    W2b = self.projectors[f"{i}_{j}"][q][1]
                    F2 += W2b(W2a(pairwise_product))

        # ✅ 2차항 누적 후 평균 (NaN/Inf 방지)
        F2 = F2 / (len(self.shift_offsets) ** 2)  # 81로 나누기

        # ✅ 최종 출력
        out = F1 + F2

        # NaN, INF 체크 (디버깅용)
        if torch.isnan(out).any():
            print("❗ NaN detected in output!")
        if torch.isinf(out).any():
            print("❗ Inf detected in output!")

        return out
