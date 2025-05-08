# reference 이미지 (깨끗한 원본) vs 복원된 이미지

""" import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from mr_vnet_model.mrvnet_unet import MRVNetUNet

# ✅ 설정
checkpoint_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid\kadid_mrvnet_epoch98.pth"
input_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I11_19_05.png"  # 왜곡 이미지
ref_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I11.png"  # ✅ 원본 Reference 이미지
output_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\single_results\result22.png"
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 불러오기
model = MRVNetUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 입력 이미지 로드 (왜곡된 이미지)
input_image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# ✅ Reference 이미지 로드 (깨끗한 원본)
ref_image = Image.open(ref_image_path).convert("RGB")
ref_tensor = transform(ref_image)

# ✅ 복원 수행
with torch.no_grad():
    output_tensor = model(input_tensor).squeeze(0).clamp(0, 1).cpu()

# ✅ 시각화 및 저장
def visualize_and_save(input_img, output_img, save_path):
    from torchvision.utils import make_grid
    import numpy as np

    grid = make_grid([input_img, output_img], nrow=2)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title('Blurred / Restored')
    plt.savefig(save_path)
    plt.close()

visualize_and_save(input_tensor.squeeze(0).cpu(), output_tensor, output_image_path)
print(f"✅ 복원 결과 저장 완료: {output_image_path}")

# ✅ PSNR, SSIM 계산 (Reference vs 복원된 이미지)
ref_np = ref_tensor.permute(1, 2, 0).cpu().numpy()
output_np = output_tensor.permute(1, 2, 0).cpu().numpy()

psnr_value = compare_psnr(ref_np, output_np, data_range=1.0)
ssim_value = compare_ssim(ref_np, output_np, channel_axis=-1, data_range=1.0)

print(f"✅ PSNR (Ref vs Restored): {psnr_value:.2f} dB")
print(f"✅ SSIM (Ref vs Restored): {ssim_value:.4f}") """
 


# 왜곡 이미지와 vs 그 왜곡을 복원한 이미지 -> 얼마나 복원햇는가
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

# ✅ 설정
checkpoint_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid\kadid_mrvnet_epoch98.pth"
input_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I11_19_05.png"  # 왜곡 이미지
output_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\single_results\result_dist_vs_restored.png"
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 불러오기
model = MRVNetUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 입력 이미지 로드 (왜곡된 이미지)
input_image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# ✅ 복원 수행
with torch.no_grad():
    output_tensor = model(input_tensor).squeeze(0).clamp(0, 1).cpu()

# ✅ 시각화 및 저장
def visualize_and_save(input_img, output_img, save_path):
    from torchvision.utils import make_grid
    import numpy as np

    grid = make_grid([input_img, output_img], nrow=2)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title('Distorted / Restored')
    plt.savefig(save_path)
    plt.close()

visualize_and_save(input_tensor.squeeze(0).cpu(), output_tensor, output_image_path)
print(f"✅ 복원 결과 저장 완료: {output_image_path}")

# ✅ PSNR, SSIM 계산 (Distorted vs Restored)
dist_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
restored_np = output_tensor.permute(1, 2, 0).cpu().numpy()

psnr_value = compare_psnr(dist_np, restored_np, data_range=1.0)
ssim_value = compare_ssim(dist_np, restored_np, channel_axis=-1, data_range=1.0)

print(f"✅ PSNR (Distorted vs Restored): {psnr_value:.2f} dB")
print(f"✅ SSIM (Distorted vs Restored): {ssim_value:.4f}")


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
checkpoint_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid\kadid_mrvnet_epoch98.pth"
distorted_img_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I11_19_05.png"
reference_img_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I11.png"
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
