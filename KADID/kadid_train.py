import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import lpips
import numpy as np
import random
import cv2
from PIL import Image
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 하이퍼파라미터 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
BASE_CHANNELS = 32
RANK = 4
LPIPS_WEIGHT = 0.1

# ✅ 경로 설정
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\KADID10K\images"
CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kkkadid"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 고도화된 카메라 시나리오 기반 transform
class CustomTrainTransform:
    def __init__(self):
        self.base = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)], p=0.5),
            transforms.RandomHorizontalFlip()
        ])

    def __call__(self, img):
        if random.random() < 0.3:
            img_np = np.array(img).astype(np.float32) / 255.
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[..., 1] *= random.uniform(0.8, 1.2)
            lab[..., 2] *= random.uniform(0.8, 1.2)
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            img = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        img = self.base(img)
        return transforms.ToTensor()(img)

# ✅ 전체 데이터셋 로딩 및 분할
full_dataset = ImageRestoreDataset(csv_path=CSV_PATH, img_dir=IMG_DIR)
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
valid_len = int(0.1 * total_len)
test_len = total_len - train_len - valid_len
train_set, valid_set, test_set = random_split(full_dataset, [train_len, valid_len, test_len])

# ✅ train에만 transform 적용
train_set.dataset.transform = CustomTrainTransform()

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델 및 손실 함수
model = MRVNetUNet(in_channels=3, base_channels=BASE_CHANNELS, rank=RANK).to(DEVICE)
mse_criterion = nn.MSELoss()
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# ✅ 학습 루프
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}]")

    for dist_img, ref_img in loop:
        dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)

        # ✅ 입력 노이즈 추가 (센서 노이즈 시뮬레이션)
        dist_img = dist_img + 0.01 * torch.randn_like(dist_img)
        dist_img = torch.clamp(dist_img, 0.0, 1.0)

        optimizer.zero_grad()
        output = model(dist_img)

        output_lpips = (output * 2) - 1
        ref_lpips = (ref_img * 2) - 1

        mse_loss = mse_criterion(output, ref_img)
        lpips_loss = lpips_fn(output_lpips, ref_lpips).mean()
        loss = mse_loss + LPIPS_WEIGHT * lpips_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"[Epoch {epoch}] Avg Train Loss: {epoch_loss / len(train_loader):.4f}")

    # ✅ Validation
    model.eval()
    with torch.no_grad():
        val_psnr, val_ssim = 0, 0
        for dist_img, ref_img in valid_loader:
            dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
            output = model(dist_img)
            val_psnr += calculate_psnr(output, ref_img)
            val_ssim += calculate_ssim(output, ref_img)

        print(f"[Epoch {epoch}] Valid PSNR: {val_psnr / len(valid_loader):.2f}, SSIM: {val_ssim / len(valid_loader):.4f}")

    # ✅ 체크포인트 저장
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"re_mrvnet_epoch{epoch}.pth"))

# ✅ TTA 함수 정의
def test_with_tta(model, image):
    flips = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),
        lambda x: torch.flip(x, dims=[-2])
    ]
    outputs = []
    for f in flips:
        aug = f(image)
        out = model(aug)
        outputs.append(f(out))
    return torch.stack(outputs).mean(dim=0)

# ✅ 최종 테스트 (TTA 적용)
model.eval()
with torch.no_grad():
    test_psnr, test_ssim = 0, 0
    for dist_img, ref_img in test_loader:
        dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
        output = test_with_tta(model, dist_img)
        test_psnr += calculate_psnr(output, ref_img)
        test_ssim += calculate_ssim(output, ref_img)

    print(f"[Final Test] PSNR: {test_psnr / len(test_loader):.2f}, SSIM: {test_ssim / len(test_loader):.4f}")
