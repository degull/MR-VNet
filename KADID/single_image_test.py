
# 출력예시
# 🔍 PSNR / SSIM 비교 결과
# Reference vs Distorted → PSNR: 18.32 dB, SSIM: 0.7112
# Reference vs Restored  → PSNR: 31.08 dB, SSIM: 0.9185
# Distorted vs Restored  → PSNR: 22.87 dB, SSIM: 0.8117

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from mr_vnet_model.mrvnet_unet import MRVNetUNet

# ✅ 설정 경로
checkpoint_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_split\re_mrvnet_epoch67.pth"
distorted_img_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\i02_11_5.bmp"
reference_img_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\I02.BMP"
save_img_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\single_results\compare_full.png"
os.makedirs(os.path.dirname(save_img_path), exist_ok=True)

# ✅ 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
model = MRVNetUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 이미지 로드
dist_img = Image.open(distorted_img_path).convert("RGB")
ref_img = Image.open(reference_img_path).convert("RGB")

dist_tensor = transform(dist_img).unsqueeze(0).to(device)
ref_tensor = transform(ref_img)

# ✅ 복원 수행
with torch.no_grad():
    restored_tensor = model(dist_tensor).squeeze(0).clamp(0, 1).cpu()

# ✅ 시각화 저장
def visualize_all(ref, dist, restored, save_path):
    from torchvision.utils import make_grid
    import numpy as np

    grid = make_grid([ref, dist, restored], nrow=3, padding=10)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title("Reference / Distorted / Restored")
    plt.savefig(save_path)
    plt.close()

visualize_all(ref_tensor, dist_tensor.squeeze(0).cpu(), restored_tensor, save_img_path)
print(f"✅ 이미지 저장 완료: {save_img_path}")

# ✅ 수치 계산
ref_np = ref_tensor.permute(1, 2, 0).cpu().numpy()
dist_np = dist_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
restored_np = restored_tensor.permute(1, 2, 0).cpu().numpy()

psnr_ref_dist = compare_psnr(ref_np, dist_np, data_range=1.0)
ssim_ref_dist = compare_ssim(ref_np, dist_np, channel_axis=-1, data_range=1.0)

psnr_ref_restored = compare_psnr(ref_np, restored_np, data_range=1.0)
ssim_ref_restored = compare_ssim(ref_np, restored_np, channel_axis=-1, data_range=1.0)

psnr_dist_restored = compare_psnr(dist_np, restored_np, data_range=1.0)
ssim_dist_restored = compare_ssim(dist_np, restored_np, channel_axis=-1, data_range=1.0)

# ✅ 출력
print("\n🔍 PSNR / SSIM 비교 결과")
print(f"Reference vs Distorted → PSNR: {psnr_ref_dist:.2f} dB, SSIM: {ssim_ref_dist:.4f}")
print(f"Reference vs Restored  → PSNR: {psnr_ref_restored:.2f} dB, SSIM: {ssim_ref_restored:.4f}")
print(f"Distorted vs Restored  → PSNR: {psnr_dist_restored:.2f} dB, SSIM: {ssim_dist_restored:.4f}")
