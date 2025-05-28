# test.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
RANK = 4
USE_LOSSLESS = True
USE_SWIN_MIDDLE = True  # ✅ Middle Block에 SwinIR + Volterra 사용

# ✅ 경로
TEST_CSV = r"E:\MRVNet2D\KADID10K\kadid10k.csv"
IMAGE_ROOT = r"E:\MRVNet2D\KADID10K\images"
MODEL_PATH = r"E:\MRVNet2D\checkpoints\swin_volterra\mrvnet_epoch100.pth"  # 원하는 에포크

# ✅ 데이터셋 로드
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
test_dataset = ImageRestoreDataset(csv_path=TEST_CSV, img_dir=IMAGE_ROOT, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ✅ 모델 초기화 및 로드
model = MRVNetUNet(
    in_channels=3,
    base_channels=32,
    rank=RANK,
    use_lossless=USE_LOSSLESS,
    use_swin_middle=USE_SWIN_MIDDLE
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# ✅ 테스트 수행
total_psnr, total_ssim = 0.0, 0.0

with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)

            for i in range(outputs.size(0)):
                psnr = calculate_psnr(outputs[i], targets[i])
                ssim = calculate_ssim(outputs[i], targets[i])
                total_psnr += psnr
                total_ssim += ssim

            pbar.set_postfix(PSNR=psnr, SSIM=ssim)
            pbar.update(1)

avg_psnr = total_psnr / len(test_dataset)
avg_ssim = total_ssim / len(test_dataset)

print(f"\n📊 [RESULT] Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")
