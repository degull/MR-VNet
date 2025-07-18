import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from mr_vnet_model.dataset_motionblur import GoProDataset
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = r'E:\MRVNet2D\dataset\GOPRO'
    csv_file = 'gopro_train_pairs.csv'
    save_dir = r'./checkpoints/gopro'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = GoProDataset(dataset_path, csv_file)
    print(f"[INFO] Found {len(train_dataset)} training samples")

    # OOM 방지
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    model = MRVNetUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()

    num_epochs = 100
    total_iters = len(train_loader)

    psnr_per_epoch = []
    ssim_per_epoch = []

    overall_start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\n[INFO] Starting Epoch {epoch+1}/{num_epochs}")
        model.train()
        total_loss = 0
        start_time = time.time()

        for idx, (blur, sharp) in enumerate(train_loader):
            iter_start_time = time.time()

            blur, sharp = blur.to(device), sharp.to(device)
            restored = model(blur)
            loss = criterion(restored, sharp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            blur_np = tensor_to_numpy(blur[0])
            gt_np = tensor_to_numpy(sharp[0])
            out_np = tensor_to_numpy(restored[0])

            psnr = compare_psnr(gt_np, out_np, data_range=255)
            ssim = compare_ssim(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)

            iter_time = time.time() - iter_start_time
            remain_iters = total_iters - (idx + 1)
            remain_time = remain_iters * iter_time
            remain_time_str = time.strftime('%H:%M:%S', time.gmtime(remain_time))

            progress = (idx + 1) / total_iters * 100

            elapsed_training_time = time.time() - overall_start_time
            elapsed_training_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_training_time))

            print(f"[Epoch {epoch+1}] Iter {idx+1}/{total_iters}  "
                  f"PSNR: {psnr:.2f}  SSIM: {ssim:.4f}  "
                  f"Progress: {progress:.1f}%  ETA: {remain_time_str}  "
                  f"Elapsed: {elapsed_training_time_str}")

        avg_loss = total_loss / len(train_loader)
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))

        print(f"[GOPRO] Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.4f}  Elapsed: {elapsed_str}")

        model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for blur, sharp in train_loader:
                blur, sharp = blur.to(device), sharp.to(device)
                restored = model(blur)

                for i in range(blur.shape[0]):
                    gt_np = tensor_to_numpy(sharp[i])
                    out_np = tensor_to_numpy(restored[i])

                    psnr = compare_psnr(gt_np, out_np, data_range=255)
                    ssim = compare_ssim(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)

                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)

        psnr_per_epoch.append(avg_psnr)
        ssim_per_epoch.append(avg_ssim)

        print(f"\n[Epoch {epoch+1}] Average PSNR: {avg_psnr:.2f} dB  Average SSIM: {avg_ssim:.4f}")

        filename = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(save_dir, filename))

    print("\n[INFO] All Epoch PSNR / SSIM Results:")
    for i in range(num_epochs):
        print(f"Epoch {i+1}: PSNR = {psnr_per_epoch[i]:.2f} dB, SSIM = {ssim_per_epoch[i]:.4f}")


if __name__ == "__main__":
    main()
