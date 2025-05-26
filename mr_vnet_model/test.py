import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset import RestorationDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
RANK = 4
USE_LOSSLESS = True  # 학습 때 사용한 것과 동일하게 설정
CHECKPOINT_PATH = './checkpoints/mrvnet_epoch100.pth'

# ✅ 경로
TEST_CSV = 'path/to/test.csv'
IMAGE_ROOT = 'path/to/images'

# ✅ 데이터 로더
transform = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = RestorationDataset(csv_file=TEST_CSV, root_dir=IMAGE_ROOT, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ✅ 모델 로딩
model = MRVNetUNet(in_channels=3, base_channels=32, rank=RANK, use_lossless=USE_LOSSLESS).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {CHECKPOINT_PATH}")

# ✅ 평가 지표 초기화
total_psnr = 0
total_ssim = 0

with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            psnr = calculate_psnr(outputs, targets)
            ssim = calculate_ssim(outputs, targets)

            total_psnr += psnr
            total_ssim += ssim
            pbar.set_postfix(PSNR=psnr, SSIM=ssim)
            pbar.update(1)

# ✅ 최종 결과
avg_psnr = total_psnr / len(test_loader)
avg_ssim = total_ssim / len(test_loader)
print(f"\n📊 [RESULT] Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")
