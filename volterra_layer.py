# volterra_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VolterraLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, rank=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.rank = rank
        self.in_channels = in_channels
        self.out_channels = out_channels

        # ✅ 1차 항: 일반 CNN처럼 선형 컨볼루션
        # → 2️⃣ 특징 추출 단계에서 가장 기본적인 색/모양 정보 추출
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)

        # ✅ 2차 항: 비선형 컨볼루션 조합을 위한 필터쌍 정의
        # → 각각 다른 필터로 conv한 뒤 곱해서 복잡한 왜곡 패턴 추출 (2️⃣의 핵심)
        self.W2a = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            for _ in range(rank)
        ])
        self.W2b = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            for _ in range(rank)
        ])

        self._init_weights()

    def _init_weights(self):
        # 초기값을 작게 설정하여 폭발 방지
        nn.init.kaiming_normal_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='linear')
        for conv in self.W2a:
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')
        for conv in self.W2b:
            nn.init.kaiming_normal_(conv.weight, a=0, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        # ✅ 선형 필터 (1차항) → 기본 구조, 윤곽 등 추출
        linear_term = self.conv1(x) # 2️⃣ enc1 ~ enc4의 선형 정보 담당

        quadratic_term = 0
        for a, b in zip(self.W2a, self.W2b):
            # ✅ 각각 다른 필터로 conv 수행 → 비선형 정보의 재료
            qa = a(x)   # 1st convolution -> conv2d 적용
            qb = b(x)   # 2nd convolution -> 또 다른 conv2d 적용

            # 안정화: 값 클램핑 (또는 normalization 해도 됨)
            qa = torch.clamp(qa, min=-1.0, max=1.0)
            qb = torch.clamp(qb, min=-1.0, max=1.0)

            # ✅ 비선형 정보: 두 conv 결과를 곱함 → 2차 항
            # → 곱셈을 통해 Blur, Ringing 같은 복잡한 왜곡 인식 가능
            prod = qa * qb  

            # NaN 방지 확인용 디버깅
            if torch.isnan(prod).any():
                print("❗ NaN 발생: qa * qb")
            quadratic_term += prod

        # ✅ 최종 출력 = 선형 + 비선형 정보 조합
        # → 4️⃣ Decoder에서도 동일한 방식으로 복원 정보 생성
        out = linear_term + quadratic_term

        # 최종 출력 안정성 확인
        if torch.isnan(out).any():
            print("❗ 출력에 NaN 존재")
        if torch.isinf(out).any():
            print("❗ 출력에 Inf 존재")

        return out
