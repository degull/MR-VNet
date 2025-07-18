import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from mr_vnet_model.dataset_derain import RainDataset
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

    dataset_path = r'E:\MRVNet2D\dataset\rain100L'
    save_dir = r'./checkpoints/rain100L'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = RainDataset(dataset_path, mode='train')
    print(f"[INFO] Found {len(train_dataset)} training samples")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

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

        for idx, (rain, norain) in enumerate(train_loader):
            iter_start_time = time.time()

            rain, norain = rain.to(device), norain.to(device)
            restored = model(rain)
            loss = criterion(restored, norain)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            rain_np = tensor_to_numpy(rain[0])
            gt_np = tensor_to_numpy(norain[0])
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

        print(f"[Rain100L] Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.4f}  Elapsed: {elapsed_str}")

        model.eval()
        psnr_list = []
        ssim_list = []

        with torch.no_grad():
            for rain, norain in train_loader:
                rain, norain = rain.to(device), norain.to(device)
                restored = model(rain)

                for i in range(rain.shape[0]):
                    gt_np = tensor_to_numpy(norain[i])
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

"""
[INFO] All Epoch PSNR / SSIM Results:
Epoch 1: PSNR = 12.85 dB, SSIM = 0.1574
Epoch 2: PSNR = 16.66 dB, SSIM = 0.3807
Epoch 3: PSNR = 20.20 dB, SSIM = 0.5820
Epoch 4: PSNR = 21.88 dB, SSIM = 0.6731
Epoch 5: PSNR = 23.22 dB, SSIM = 0.7198
Epoch 6: PSNR = 24.04 dB, SSIM = 0.7496
Epoch 7: PSNR = 24.75 dB, SSIM = 0.7723
Epoch 8: PSNR = 24.80 dB, SSIM = 0.7845
Epoch 9: PSNR = 25.59 dB, SSIM = 0.7989
Epoch 10: PSNR = 25.65 dB, SSIM = 0.8060
Epoch 11: PSNR = 25.82 dB, SSIM = 0.8140
Epoch 12: PSNR = 25.84 dB, SSIM = 0.8187
Epoch 13: PSNR = 26.04 dB, SSIM = 0.8251
Epoch 14: PSNR = 26.78 dB, SSIM = 0.8324
Epoch 15: PSNR = 27.00 dB, SSIM = 0.8380
Epoch 16: PSNR = 26.47 dB, SSIM = 0.8393
Epoch 17: PSNR = 27.24 dB, SSIM = 0.8423
Epoch 18: PSNR = 26.43 dB, SSIM = 0.8438
Epoch 19: PSNR = 27.55 dB, SSIM = 0.8497
Epoch 20: PSNR = 27.62 dB, SSIM = 0.8527
Epoch 21: PSNR = 27.07 dB, SSIM = 0.8527
Epoch 22: PSNR = 27.31 dB, SSIM = 0.8557
Epoch 23: PSNR = 27.73 dB, SSIM = 0.8589
Epoch 24: PSNR = 27.79 dB, SSIM = 0.8618
Epoch 25: PSNR = 27.98 dB, SSIM = 0.8636
Epoch 26: PSNR = 28.27 dB, SSIM = 0.8661
Epoch 27: PSNR = 28.44 dB, SSIM = 0.8679
Epoch 28: PSNR = 28.40 dB, SSIM = 0.8699
Epoch 29: PSNR = 28.47 dB, SSIM = 0.8708
Epoch 30: PSNR = 27.46 dB, SSIM = 0.8699
Epoch 31: PSNR = 28.18 dB, SSIM = 0.8734
Epoch 32: PSNR = 28.51 dB, SSIM = 0.8742
Epoch 33: PSNR = 28.70 dB, SSIM = 0.8765
Epoch 34: PSNR = 28.73 dB, SSIM = 0.8775
Epoch 35: PSNR = 28.83 dB, SSIM = 0.8794
Epoch 36: PSNR = 28.61 dB, SSIM = 0.8806
Epoch 37: PSNR = 28.96 dB, SSIM = 0.8818
Epoch 38: PSNR = 28.80 dB, SSIM = 0.8817
Epoch 39: PSNR = 28.40 dB, SSIM = 0.8833
Epoch 40: PSNR = 29.26 dB, SSIM = 0.8850
Epoch 41: PSNR = 29.13 dB, SSIM = 0.8874
Epoch 42: PSNR = 29.18 dB, SSIM = 0.8886
Epoch 43: PSNR = 29.23 dB, SSIM = 0.8902
Epoch 44: PSNR = 29.22 dB, SSIM = 0.8907
Epoch 45: PSNR = 29.74 dB, SSIM = 0.8935
Epoch 46: PSNR = 29.90 dB, SSIM = 0.8959
Epoch 47: PSNR = 29.38 dB, SSIM = 0.8983
Epoch 48: PSNR = 29.01 dB, SSIM = 0.8958
Epoch 49: PSNR = 29.65 dB, SSIM = 0.9001
Epoch 50: PSNR = 30.29 dB, SSIM = 0.9031
Epoch 51: PSNR = 30.39 dB, SSIM = 0.9066
Epoch 52: PSNR = 30.93 dB, SSIM = 0.9096
Epoch 53: PSNR = 29.41 dB, SSIM = 0.9109
Epoch 54: PSNR = 31.30 dB, SSIM = 0.9158
Epoch 55: PSNR = 31.52 dB, SSIM = 0.9191
Epoch 56: PSNR = 31.19 dB, SSIM = 0.9204
Epoch 57: PSNR = 31.93 dB, SSIM = 0.9245
Epoch 58: PSNR = 31.66 dB, SSIM = 0.9251
Epoch 59: PSNR = 32.21 dB, SSIM = 0.9272
Epoch 60: PSNR = 31.55 dB, SSIM = 0.9298
Epoch 61: PSNR = 32.51 dB, SSIM = 0.9317
Epoch 62: PSNR = 32.27 dB, SSIM = 0.9367
Epoch 63: PSNR = 32.83 dB, SSIM = 0.9378
Epoch 64: PSNR = 33.07 dB, SSIM = 0.9401
Epoch 65: PSNR = 33.26 dB, SSIM = 0.9418
Epoch 66: PSNR = 33.70 dB, SSIM = 0.9451
Epoch 67: PSNR = 32.52 dB, SSIM = 0.9444
Epoch 68: PSNR = 33.33 dB, SSIM = 0.9462
Epoch 69: PSNR = 33.83 dB, SSIM = 0.9486
Epoch 70: PSNR = 33.89 dB, SSIM = 0.9503
Epoch 71: PSNR = 33.66 dB, SSIM = 0.9497
Epoch 72: PSNR = 33.68 dB, SSIM = 0.9507
Epoch 73: PSNR = 33.57 dB, SSIM = 0.9535
Epoch 74: PSNR = 34.20 dB, SSIM = 0.9528
Epoch 75: PSNR = 33.80 dB, SSIM = 0.9542
Epoch 76: PSNR = 34.55 dB, SSIM = 0.9557
Epoch 77: PSNR = 34.25 dB, SSIM = 0.9579
Epoch 78: PSNR = 34.82 dB, SSIM = 0.9585
Epoch 79: PSNR = 35.42 dB, SSIM = 0.9617
Epoch 80: PSNR = 34.27 dB, SSIM = 0.9609
Epoch 81: PSNR = 35.34 dB, SSIM = 0.9625
Epoch 82: PSNR = 35.41 dB, SSIM = 0.9631
Epoch 83: PSNR = 35.18 dB, SSIM = 0.9636
Epoch 84: PSNR = 35.52 dB, SSIM = 0.9649
Epoch 85: PSNR = 35.38 dB, SSIM = 0.9639
Epoch 86: PSNR = 35.72 dB, SSIM = 0.9653
Epoch 87: PSNR = 35.70 dB, SSIM = 0.9678
Epoch 88: PSNR = 35.58 dB, SSIM = 0.9645
Epoch 89: PSNR = 36.23 dB, SSIM = 0.9684
Epoch 90: PSNR = 35.37 dB, SSIM = 0.9683
Epoch 91: PSNR = 35.58 dB, SSIM = 0.9692
Epoch 92: PSNR = 35.97 dB, SSIM = 0.9700
Epoch 93: PSNR = 35.69 dB, SSIM = 0.9717
Epoch 94: PSNR = 35.88 dB, SSIM = 0.9719
Epoch 95: PSNR = 35.98 dB, SSIM = 0.9720
Epoch 96: PSNR = 36.20 dB, SSIM = 0.9705
Epoch 97: PSNR = 35.93 dB, SSIM = 0.9710
Epoch 98: PSNR = 36.58 dB, SSIM = 0.9715
Epoch 99: PSNR = 36.07 dB, SSIM = 0.9726
Epoch 100: PSNR = 37.38 dB, SSIM = 0.9749
"""