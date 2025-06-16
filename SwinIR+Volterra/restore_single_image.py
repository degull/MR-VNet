import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from mrvnet_unet import MRVNetUNet
from utils import calculate_psnr, calculate_ssim

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = 4
USE_LOSSLESS = True
USE_SWIN_MIDDLE = True

# ✅ 경로 설정
DISTORTED_IMAGE_PATH = r"E:\MRVNet2D\dataset\KADID10K\images\I20_02_05.png"   # 입력 왜곡 이미지
REFERENCE_IMAGE_PATH = r"E:\MRVNet2D\dataset\KADID10K\images\I20.png"        # 참조 이미지 (ground truth)
MODEL_PATH = r"E:\MRVNet2D\checkpoints\volterra_swinir_hybrid\mrvnet_epoch100.pth"

# ✅ 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# ✅ 이미지 로딩 및 전처리
input_img = transform(Image.open(DISTORTED_IMAGE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)
ref_img = transform(Image.open(REFERENCE_IMAGE_PATH).convert("RGB")).unsqueeze(0).to(DEVICE)

# ✅ 모델 로드
model = MRVNetUNet(
    in_channels=3,
    base_channels=32,
    rank=RANK,
    use_lossless=USE_LOSSLESS,
    use_swin_middle=USE_SWIN_MIDDLE
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    restored = model(input_img)

# ✅ PSNR / SSIM 계산
psnr = calculate_psnr(restored[0], ref_img[0])
ssim = calculate_ssim(restored[0], ref_img[0])

# ✅ 시각화
input_pil = to_pil(input_img.squeeze().cpu())
restored_pil = to_pil(restored.squeeze().clamp(0, 1).cpu())
ref_pil = to_pil(ref_img.squeeze().cpu())

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(input_pil)
axs[0].set_title("Distorted Image")
axs[1].imshow(restored_pil)
axs[1].set_title("Restored Image")
axs[2].imshow(ref_pil)
axs[2].set_title("Reference Image")

for ax in axs:
    ax.axis("off")

plt.suptitle(f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}", fontsize=14)
plt.tight_layout()
plt.show()
