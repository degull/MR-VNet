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

        # ✅ shift offset 목록 생성
        self.shift_offsets = [
            (dy, dx)
            for dy in range(-shift_radius, shift_radius + 1)
            for dx in range(-shift_radius, shift_radius + 1)
        ]

        # ✅ 2차항: 공유된 rank 필터 사용
        self.W2a = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in range(rank)
        ])
        self.W2b = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(rank)
        ])

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.linear_conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        for conv in self.W2a:
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        for conv in self.W2b:
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')

    def circular_shift(self, x, shift):
        return torch.roll(x, shifts=shift, dims=(2, 3))

    def forward(self, x):
        # ✅ 1차항
        F1 = self.linear_conv(x)

        # ✅ 2차항
        shifted_maps = [self.circular_shift(x, shift) for shift in self.shift_offsets]
        F2 = 0
        count = 0

        for i, Xi in enumerate(shifted_maps):
            for j, Xj in enumerate(shifted_maps):
                if j < i:
                    continue  # 대칭 중복 제거
                pairwise_product = Xi * Xj
                for q in range(self.rank):
                    proj = self.W2b[q](self.W2a[q](pairwise_product))
                    F2 += proj
                count += 1

        F2 = F2 / count  # 평균 내기

        return F1 + F2
