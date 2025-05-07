# test.py

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import lpips

# 📦 로컬 모듈 임포트
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kadid_volterra_model.mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

def test():
    # ✅ 설정
    CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
    IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"
    MODEL_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\kadid_volterras\mrvnet_epoch3.pth"
    SAVE_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1  # 테스트는 1장씩

    # ✅ 데이터 로딩
    test_dataset = ImageRestoreDataset(csv_path=CSV_PATH, img_dir=IMG_DIR)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ✅ 모델 로딩
    model = MRVNetUNet(in_channels=3, base_channels=32, shift_radius=1, rank=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ✅ LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

    # ✅ 평가 지표 누적 변수
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0

    # ✅ 평가 루프
    with torch.no_grad():
        for idx, (dist_img, ref_img) in enumerate(tqdm(test_loader, desc="Evaluating")):
            dist_img = dist_img.to(DEVICE)
            ref_img = ref_img.to(DEVICE)

            output = model(dist_img)

            # 점수 계산
            psnr = calculate_psnr(output, ref_img)
            ssim = calculate_ssim(output, ref_img)
            lpips_score = lpips_fn((output * 2) - 1, (ref_img * 2) - 1).mean().item()

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score
            count += 1

            # ✅ 시각화 저장 (원본/복원)
            save_image(dist_img, os.path.join(SAVE_DIR, f"{idx:04d}_input.png"))
            save_image(output, os.path.join(SAVE_DIR, f"{idx:04d}_restored.png"))
            save_image(ref_img, os.path.join(SAVE_DIR, f"{idx:04d}_gt.png"))

    # ✅ 평균 점수 출력
    print(f"\n✅ [Test Results]")
    print(f"Average PSNR : {total_psnr / count:.2f} dB")
    print(f"Average SSIM : {total_ssim / count:.4f}")
    print(f"Average LPIPS: {total_lpips / count:.4f}")

if __name__ == "__main__":
    test()
