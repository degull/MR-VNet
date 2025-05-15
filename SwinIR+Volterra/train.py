import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from swinir_volterra import SwinIRVolterra
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 하이퍼파라미터
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_swinir"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 데이터셋 경로
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"

# ✅ 데이터셋 로딩 및 분할
transform = transforms.ToTensor()
full_dataset = ImageRestoreDataset(CSV_PATH, IMG_DIR, transform=transform)

total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len
train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# ✅ 모델, 손실함수, 최적화
model = SwinIRVolterra().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for distorted, reference in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
        output = model(distorted)

        loss = criterion(output, reference)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # ✅ 검증
    model.eval()
    total_psnr, total_ssim = 0.0, 0.0
    with torch.no_grad():
        for distorted, reference in val_loader:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            output = model(distorted)
            total_psnr += calculate_psnr(output, reference)
            total_ssim += calculate_ssim(output, reference)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)

    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

    # ✅ 체크포인트 저장
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"swinir_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), checkpoint_path)
