# test.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from mrvnet_unet import MRVNetUNet
from dataset import ImageRestoreDataset
from utils import calculate_psnr, calculate_ssim

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\kadid10k.csv"
IMG_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images"
CHECKPOINT = r"C:/Users/IIPL02/Desktop/MRVNet2D/checkpoints/re_mrvnet_epoch98.pth"
RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ✅ 데이터셋 및 모델
test_dataset = ImageRestoreDataset(csv_path=CSV_PATH, img_dir=IMG_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = MRVNetUNet(in_channels=3, base_channels=32, rank=4).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

# ✅ 결과 저장용 변수
all_loss, all_psnr, all_ssim, all_lpips = [], [], [], []

with torch.no_grad():
    for idx, (dist_img, ref_img) in enumerate(tqdm(test_loader, desc="Testing")):
        dist_img = dist_img.to(DEVICE)
        ref_img = ref_img.to(DEVICE)

        output = model(dist_img)

        # ✅ 수치 계산
        loss = torch.nn.functional.mse_loss(output, ref_img).item()
        psnr = calculate_psnr(output, ref_img)
        ssim = calculate_ssim(output, ref_img)
        lpips_score = lpips_fn(output, ref_img).item()

        all_loss.append(loss)
        all_psnr.append(psnr)
        all_ssim.append(ssim)
        all_lpips.append(lpips_score)

        # ✅ 시각화 저장
        if idx < 10:  # 처음 10개만 저장
            save_image(dist_img, f"{RESULT_DIR}/input_{idx:03d}.png")
            save_image(ref_img, f"{RESULT_DIR}/ref_{idx:03d}.png")
            save_image(output, f"{RESULT_DIR}/output_{idx:03d}.png")

            compare = torch.cat([dist_img, output, ref_img], dim=3)
            save_image(compare, f"{RESULT_DIR}/compare_{idx:03d}.png")

# ✅ 그래프 저장
def save_plot(values, title, ylabel, filename):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, filename))
    plt.close()

save_plot(all_loss, "MSE Loss per Image", "MSE Loss", "loss.png")
save_plot(all_psnr, "PSNR per Image", "PSNR (dB)", "psnr.png")
save_plot(all_ssim, "SSIM per Image", "SSIM", "ssim.png")
save_plot(all_lpips, "LPIPS per Image", "LPIPS", "lpips.png")

# ✅ 평균 출력
print(f"\n📊 전체 평균:")
print(f"Loss: {sum(all_loss)/len(all_loss):.4f}")
print(f"PSNR: {sum(all_psnr)/len(all_psnr):.2f}")
print(f"SSIM: {sum(all_ssim)/len(all_ssim):.4f}")
print(f"LPIPS: {sum(all_lpips)/len(all_lpips):.4f}")
