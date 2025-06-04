# 전체 데이터셋 복원
""" 
import os
import torch
from torchvision import transforms
from PIL import Image
from restormer_volterra import RestormerVolterra  # 동일한 모델 정의 파일
from kadid_dataset import KADID10KDataset         # 동일한 커스텀 데이터셋

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'E:/MRVNet2D/checkpoints/restormer_volterra_kadid/epoch_100.pth'
DATA_CSV = 'E:/MRVNet2D/dataset/KADID10K/kadid10k.csv'
IMAGE_DIR = 'E:/MRVNet2D/dataset/KADID10K/images'
SAVE_DIR = 'E:/MRVNet2D/results/restored_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 데이터셋 준비
test_dataset = KADID10KDataset(csv_file=DATA_CSV, transform=transform)

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    for idx in range(len(test_dataset)):
        distorted, reference = test_dataset[idx]  # (tensor, tensor)
        input_img = distorted.unsqueeze(0).to(DEVICE)

        output = model(input_img)
        output_img = output.squeeze(0).cpu().clamp(0, 1)

        # 저장
        restored_pil = transforms.ToPILImage()(output_img)
        restored_pil.save(os.path.join(SAVE_DIR, f"restored_{idx:05d}.png"))

        if idx < 5:  # ✅ 처음 몇 개만 확인용 출력
            print(f"Saved: restored_{idx:05d}.png")

print("✅ 테스트 이미지 복원 완료.")
 """

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from restormer_volterra import RestormerVolterra
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np

# ✅ 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'E:/MRVNet2D/checkpoints/restormer_volterra_kadid/epoch_100.pth'
DISTORTED_IMG_PATH = 'E:/MRVNet2D/tid2013/distorted_images/i01_04_5.bmp'    # 복원할 이미지
REFERENCE_IMG_PATH = 'E:/MRVNet2D/tid2013/reference_images/I01.BMP'    # 정답 이미지
SAVE_PATH = 'E:/MRVNet2D/results/comparison_result.png'

# ✅ 변환 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
to_pil = transforms.ToPILImage()

# ✅ 이미지 로드
distorted = Image.open(DISTORTED_IMG_PATH).convert('RGB')
reference = Image.open(REFERENCE_IMG_PATH).convert('RGB')

distorted_tensor = transform(distorted).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
reference_tensor = transform(reference).unsqueeze(0).to(DEVICE)

# ✅ 모델 로드
model = RestormerVolterra().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    output_tensor = model(distorted_tensor).clamp(0, 1)

# ✅ PSNR, SSIM 계산
output_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
reference_np = reference_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()


psnr = compute_psnr(reference_np, output_np, data_range=1.0)
ssim = compute_ssim(reference_np, output_np, data_range=1.0, channel_axis=2, win_size=7)

print(f"✅ PSNR: {psnr:.2f} dB")
print(f"✅ SSIM: {ssim:.4f}")

# ✅ 이미지 붙이기 및 저장
ref_img = to_pil(reference_tensor.squeeze(0).cpu())
dist_img = to_pil(distorted_tensor.squeeze(0).cpu())
out_img = to_pil(output_tensor.squeeze(0).cpu())

width, height = ref_img.size
concat_img = Image.new('RGB', (width * 3, height))
concat_img.paste(ref_img, (0, 0))
concat_img.paste(dist_img, (width, 0))
concat_img.paste(out_img, (width * 2, 0))
concat_img.save(SAVE_PATH)

print(f"✅ 비교 이미지 저장 완료: {SAVE_PATH}")
