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

def main():
    # ✅ 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RANK = 4
    USE_LOSSLESS = True  # 학습 설정과 동일하게 유지
    CHECKPOINT_PATH = r"E:\MRVNet2D\checkpoints\original_volterra\mrvnet_epoch97.pth"

    # ✅ 테스트 데이터 경로
    TEST_CSV = r"E:\MRVNet2D\KADID10K\kadid10k.csv"
    IMAGE_ROOT = r"E:\MRVNet2D\KADID10K\images"

    # ✅ Transform 정의
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # ✅ Dataset & Dataloader
    test_dataset = ImageRestoreDataset(csv_path=TEST_CSV, img_dir=IMAGE_ROOT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # ✅ 모델 로딩
    model = MRVNetUNet(in_channels=3, base_channels=32, rank=RANK, use_lossless=USE_LOSSLESS).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Model loaded from {CHECKPOINT_PATH}")

    total_psnr = 0.0
    total_ssim = 0.0

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

    avg_psnr = total_psnr / len(test_loader)
    avg_ssim = total_ssim / len(test_loader)
    print(f"\n📊 [RESULT] Avg PSNR: {avg_psnr:.2f} dB, Avg SSIM: {avg_ssim:.4f}")

# ✅ Windows에서 필수
if __name__ == "__main__":
    main()
