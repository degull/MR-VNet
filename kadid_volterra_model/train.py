# train.py

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import lpips

# 🔥 MRVNet Model Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kadid_volterra_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim


def main():
    # ✅ 경로 설정
    CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
    IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"
    CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_volterras"
    PRETRAINED_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_volterras\mrvnet_epoch3.pth"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ✅ 하이퍼파라미터 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    NUM_EPOCHS = 100
    START_EPOCH = 4  # ⭐ 3epoch까지 학습한 weight 이어받으니까 4부터 시작
    LEARNING_RATE = 1e-5
    BASE_CHANNELS = 32
    SHIFT_RADIUS = 1
    RANK = 4
    LPIPS_WEIGHT = 0.1

    # ✅ 데이터셋 로딩
    train_dataset = ImageRestoreDataset(csv_path=CSV_PATH, img_dir=IMG_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # ✅ 모델, 손실함수, 옵티마이저 정의
    model = MRVNetUNet(in_channels=3, base_channels=BASE_CHANNELS, shift_radius=SHIFT_RADIUS, rank=RANK).to(DEVICE)
    mse_criterion = nn.MSELoss()
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ✅ 저장된 모델 불러오기
    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded checkpoint from: {PRETRAINED_PATH}")

    # ✅ 학습 루프
    for epoch in range(START_EPOCH, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}]")
        for dist_img, ref_img in loop:
            dist_img = dist_img.to(DEVICE)
            ref_img = ref_img.to(DEVICE)

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

        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ [Epoch {epoch}] Average Loss: {avg_loss:.6f}")

        # ✅ Epoch 끝나고 평가
        model.eval()
        with torch.no_grad():
            total_psnr = 0
            total_ssim = 0
            for dist_img, ref_img in train_loader:
                dist_img = dist_img.to(DEVICE)
                ref_img = ref_img.to(DEVICE)
                output = model(dist_img)

                psnr = calculate_psnr(output, ref_img)
                ssim = calculate_ssim(output, ref_img)

                total_psnr += psnr
                total_ssim += ssim

            avg_psnr = total_psnr / len(train_loader)
            avg_ssim = total_ssim / len(train_loader)
            print(f"✅ [Epoch {epoch}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # ✅ 체크포인트 저장
        save_path = os.path.join(CHECKPOINT_DIR, f"mrvnet_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Checkpoint saved at {save_path}")

# 🔥 진짜 중요한 부분 🔥
if __name__ == "__main__":
    main()
