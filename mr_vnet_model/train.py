# Reference 이미지(ref_img)는 학습 시 정답(target)으로만 사용되고, 입력은 항상 Distorted 이미지(dist_img) 단독
# reference 이미지는 정답(target)으로만 사용
# model에는 distorted 이미지만 입력
# → 이것은 복원 모델 (Restoration Model) 의 전형적인 Supervised 학습 방식입니다.

 
# train.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

def main():
    # ✅ 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 100
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    RANK = 4
    USE_LOSSLESS = True

    # ✅ 경로
    TRAIN_CSV = r"E:\MRVNet2D\KADID10K\kadid10k.csv"
    IMAGE_ROOT = r"E:\MRVNet2D\KADID10K\images"
    CHECKPOINT_DIR = r"E:\MRVNet2D\checkpoints\original_volterra"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ✅ 데이터 로더
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_dataset = ImageRestoreDataset(csv_path=TRAIN_CSV, img_dir=IMAGE_ROOT, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # ✅ 모델
    model = MRVNetUNet(in_channels=3, base_channels=32, rank=RANK, use_lossless=USE_LOSSLESS).to(DEVICE)

    # ✅ 손실 함수 및 최적화
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ✅ 학습 루프
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for inputs, targets in train_loader:
                inputs = inputs.to(DEVICE)  # ← distorted 이미지 (입력)
                targets = targets.to(DEVICE)    # ← reference 이미지 (정답)

                outputs = model(inputs) # ← 모델에는 inputs만 들어감
                loss = criterion(outputs, targets)  # ← 정답과 비교 (MSELoss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        print(f"📌 Epoch {epoch+1} Loss: {epoch_loss/len(train_loader):.4f}")

        # ✅ 모든 에포크마다 저장
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"mrvnet_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Checkpoint saved to {ckpt_path}")

# ✅ Windows에서 필수
if __name__ == "__main__":
    main()
