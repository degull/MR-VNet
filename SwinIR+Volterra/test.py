import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from swinir_volterra import SwinIRVolterra
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 경로 설정
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"
CHECKPOINT_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_swinir\swinir_epoch100.pth"  # 마지막 epoch 기준

# ✅ 하이퍼파라미터
BATCH_SIZE = 1

# ✅ 데이터셋 및 분할
transform = transforms.ToTensor()
full_dataset = ImageRestoreDataset(CSV_PATH, IMG_DIR, transform=transform)

total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len
_, _, test_set = torch.utils.data.random_split(full_dataset, [train_len, val_len, test_len])

test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델 불러오기
model = SwinIRVolterra().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# ✅ 테스트 수행
total_psnr = 0.0
total_ssim = 0.0

with torch.no_grad():
    for distorted, reference in test_loader:
        distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
        output = model(distorted)

        total_psnr += calculate_psnr(output, reference)
        total_ssim += calculate_ssim(output, reference)

avg_psnr = total_psnr / len(test_loader)
avg_ssim = total_ssim / len(test_loader)

print(f"✅ Test PSNR: {avg_psnr:.2f} dB")
print(f"✅ Test SSIM: {avg_ssim:.4f}")
