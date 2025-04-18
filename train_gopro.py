import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import lpips

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from dataset_gopro_sidd import PairedImageDataset
from utils import calculate_psnr, calculate_ssim

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    BASE_CHANNELS = 32
    RANK = 4
    LPIPS_WEIGHT = 0.1

    CSV_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\GOPRO_Large\gopro_train_pairs.csv"
    CHECKPOINT_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\gopro"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = PairedImageDataset(csv_path=CSV_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = MRVNetUNet(in_channels=3, base_channels=BASE_CHANNELS, rank=RANK).to(DEVICE)
    mse_criterion = nn.MSELoss()
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"[Epoch {epoch}]")

        for dist_img, ref_img in loop:
            dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
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

        print(f"\n✅ [Epoch {epoch}] Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # 평가
        model.eval()
        with torch.no_grad():
            total_psnr = 0
            total_ssim = 0
            for dist_img, ref_img in train_loader:
                dist_img, ref_img = dist_img.to(DEVICE), ref_img.to(DEVICE)
                output = model(dist_img)
                psnr = calculate_psnr(output, ref_img)
                ssim = calculate_ssim(output, ref_img)
                total_psnr += psnr
                total_ssim += ssim
            print(f"📊 PSNR: {total_psnr / len(train_loader):.2f}, SSIM: {total_ssim / len(train_loader):.4f}")

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"mrvnet_epoch{epoch}.pth"))

# ✅ 반드시 추가
if __name__ == "__main__":
    main()
