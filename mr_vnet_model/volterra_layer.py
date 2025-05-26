# volterra_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def circular_shift(x, shift_x, shift_y):
    """ 입력 feature map을 (shift_x, shift_y) 만큼 원형 이동 (circular shift) """
    return torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=4, use_lossless=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_lossless = use_lossless

        # ✅ 1차항: 선형 필터 (공통)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        if self.use_lossless:
            # ✅ Lossless: shared conv + shift 목록
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            self.shifts = self._generate_shifts(kernel_size)
        else:
            # ✅ Lossy: 분리 가능한 필터쌍 (rank 개수만큼)
            self.W2a = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])
            self.W2b = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
                for _ in range(rank)
            ])

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='linear')
        if self.use_lossless:
            nn.init.kaiming_normal_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='linear')
        else:
            for conv in self.W2a:
                nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')
            for conv in self.W2b:
                nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')

    def _generate_shifts(self, k):
        """ 대칭 항 제거된 shift 쌍 생성 (e.g., 3x3 kernel → 4개 shift) """
        P = k // 2
        shifts = []
        for s1 in range(-P, P + 1):
            for s2 in range(-P, P + 1):
                if s1 == 0 and s2 == 0:
                    continue
                if (s1, s2) < (0, 0):  # 대칭 제거
                    continue
                shifts.append((s1, s2))
        return shifts

    def forward(self, x):
        # ✅ 선형항
        linear_term = self.conv1(x)

        # ✅ 비선형항
        quadratic_term = 0

        if self.use_lossless:
            for s1, s2 in self.shifts:
                x_shifted = circular_shift(x, s1, s2)
                prod = x * x_shifted
                prod = torch.clamp(prod, min=-1.0, max=1.0)  # 안정화 필수
                quadratic_term += self.conv2(prod)
        else:
            for a, b in zip(self.W2a, self.W2b):
                qa = torch.clamp(a(x), min=-1.0, max=1.0)
                qb = torch.clamp(b(x), min=-1.0, max=1.0)
                prod = qa * qb
                if torch.isnan(prod).any():
                    print("❗ NaN 발생: qa * qb")
                quadratic_term += prod

        out = linear_term + quadratic_term

        # ✅ 안정성 검사
        if torch.isnan(out).any():
            print("❗ 출력에 NaN 존재")
        if torch.isinf(out).any():
            print("❗ 출력에 Inf 존재")

        return out
