import os
import sys
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# 환경변수 설정 (실행 초반에 위치해야 효과 있음)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from swinir_volterra import SwinIRVolterra
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 하이퍼파라미터
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_swinir"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ✅ 데이터셋 경로
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"

# ✅ 데이터 로딩 및 분할
transform = transforms.ToTensor()
full_dataset = ImageRestoreDataset(CSV_PATH, IMG_DIR, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# ✅ 모델 정의 (최소 사양)
model = SwinIRVolterra(
    img_size=(32, 32),             # 해상도 축소
    patch_size=1,
    in_chans=3,
    embed_dim=32,                 # 채널 수 축소
    depths=[2, 2, 2, 2],          # 블록 개수 축소
    num_heads=[2, 2, 2, 2],
    window_size=4,                # 윈도우 축소
    mlp_ratio=2.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    upscale=1,
    img_range=1.0,
    upsampler='',
    resi_connection='1conv',
    rank=1                       # Volterra 연산 최소화
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 학습 루프
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    # 메모리 초기화
    torch.cuda.empty_cache()
    gc.collect()

    for distorted, reference in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
        output = model(distorted)

        loss = criterion(output, reference)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # ✅ 검증 PSNR/SSIM 측정
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for distorted, reference in val_loader:
            distorted, reference = distorted.to(DEVICE), reference.to(DEVICE)
            output = model(distorted)
            psnr = calculate_psnr(output, reference)
            ssim = calculate_ssim(output, reference)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    print(f"Validation PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

    # ✅ 모델 저장
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))
