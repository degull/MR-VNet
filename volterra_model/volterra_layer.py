import torch
import torch.nn as nn
import torch.nn.functional as F

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ✅ 1차 항 - 기본 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # ✅ Shift 방향 (논문 기준: 중심 + 상하좌우)
        self.shift_offsets = [
            (0, 0),    # 중심
            (-1, 0),   # 위
            (1, 0),    # 아래
            (0, -1),   # 왼쪽
            (0, 1)     # 오른쪽
        ]

        # ✅ 각 (i, j) 쌍에 대해 독립적인 1x1 projection conv 정의
        self.projectors = nn.ModuleDict()
        for i in range(len(self.shift_offsets)):
            for j in range(len(self.shift_offsets)):
                self.projectors[f"{i}_{j}"] = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def circular_shift(self, x, shift):
        return torch.roll(x, shifts=shift, dims=(2, 3))

    def forward(self, x):
        # ✅ 1차 항
        F1 = self.conv1(x)

        # ✅ shift된 feature map 생성
        shifted_maps = [self.circular_shift(x, shift) for shift in self.shift_offsets]

        # ✅ 2차 항 계산
        F2 = 0
        for i, Xi in enumerate(shifted_maps):
            for j, Xj in enumerate(shifted_maps):
                pairwise_prod = Xi * Xj                       # 곱
                proj = self.projectors[f"{i}_{j}"](pairwise_prod)  # 1x1 projection
                F2 += proj

        # ✅ 최종 출력
        return F1 + F2
