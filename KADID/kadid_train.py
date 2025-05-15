import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import lpips

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 하이퍼파라미터 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
START_EPOCH = 68  # 🔥 이어서 시작할 에폭
LEARNING_RATE = 1e-4
BASE_CHANNELS = 32
RANK = 4
LPIPS_WEIGHT = 0.1

# ✅ 경로 설정
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"
CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_split"
RESUME_PATH = os.path.join(CHECKPOINT_DIR, "re_mrvnet_epoch67.pth")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 전체 데이터셋 로딩 및 분할
full_dataset = ImageRestoreDataset(csv_path=CSV_PATH, img_dir=IMG_DIR)
total_len = len(full_dataset)
train_len = int(0.8 * total_len)
valid_len = int(0.1 * total_len)
test_len = total_len - train_len - valid_len
train_set, valid_set, test_set = random_split(full_dataset, [train_len, valid_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# ✅ 모델, 손실 함수, 옵티마이저 정의
model = MRVNetUNet(in_channels=3, base_channels=BASE_CHANNELS, rank=RANK).to(DEVICE)
mse_criterion = nn.MSELoss()
lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 이어서 학습할 경우: 체크포인트 불러오기
if os.path.exists(RESUME_PATH):
    print(f"🔄 체크포인트 로딩 중: {RESUME_PATH}")
    model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
else:
    print(f"⚠️ 체크포인트 {RESUME_PATH} 가 존재하지 않습니다. 새로 학습을 시작합니다.")

# ✅ 학습 루프
for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}]")

    for dist_img, ref_img in loop:
        dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)

        optimizer.zero_grad()
        output = model(dist_img)

        # LPIPS 정규화
        output_lpips = (output * 2) - 1
        ref_lpips = (ref_img * 2) - 1

        # 손실 계산
        mse_loss = mse_criterion(output, ref_img)
        lpips_loss = lpips_fn(output_lpips, ref_lpips).mean()
        loss = mse_loss + LPIPS_WEIGHT * lpips_loss

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch}] Avg Train Loss: {avg_loss:.4f}")

    # ✅ Validation
    model.eval()
    with torch.no_grad():
        val_psnr, val_ssim = 0, 0
        for dist_img, ref_img in valid_loader:
            dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
            output = model(dist_img)
            val_psnr += calculate_psnr(output, ref_img)
            val_ssim += calculate_ssim(output, ref_img)

        avg_val_psnr = val_psnr / len(valid_loader)
        avg_val_ssim = val_ssim / len(valid_loader)
        print(f"[Epoch {epoch}] Valid PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}")

    # ✅ 체크포인트 저장
    save_path = os.path.join(CHECKPOINT_DIR, f"re_mrvnet_epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path)

# ✅ 최종 테스트
model.eval()
with torch.no_grad():
    test_psnr, test_ssim = 0, 0
    for dist_img, ref_img in test_loader:
        dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
        output = model(dist_img)
        test_psnr += calculate_psnr(output, ref_img)
        test_ssim += calculate_ssim(output, ref_img)

    avg_test_psnr = test_psnr / len(test_loader)
    avg_test_ssim = test_ssim / len(test_loader)
    print(f"[Final Test] PSNR: {avg_test_psnr:.2f}, SSIM: {avg_test_ssim:.4f}")
