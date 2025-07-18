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

    dataset_path = r'E:\MRVNet2D\dataset\rain100H'
    save_dir = r'./checkpoints/rain100H'
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

        print(f"[Rain100H] Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.4f}  Elapsed: {elapsed_str}")

        model.eval()
        psnr_list, ssim_list = [], []

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

        # 반드시 PSNR, SSIM 계산 이후 저장
        filename = f"epoch_{epoch+1}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(save_dir, filename))


    # 모든 에포크 종료 후 전체 출력
    print("\n[INFO] All Epoch PSNR / SSIM Results:")
    for i in range(num_epochs):
        print(f"Epoch {i+1}: PSNR = {psnr_per_epoch[i]:.2f} dB, SSIM = {ssim_per_epoch[i]:.4f}")


if __name__ == "__main__":
    main()

"""
[INFO] All Epoch PSNR / SSIM Results:
Epoch 1: PSNR = 17.89 dB, SSIM = 0.3879
Epoch 2: PSNR = 18.79 dB, SSIM = 0.4747
Epoch 3: PSNR = 19.75 dB, SSIM = 0.5199
Epoch 4: PSNR = 18.51 dB, SSIM = 0.5460
Epoch 5: PSNR = 21.08 dB, SSIM = 0.5905
Epoch 6: PSNR = 21.75 dB, SSIM = 0.6308
Epoch 7: PSNR = 22.31 dB, SSIM = 0.6568
Epoch 8: PSNR = 21.36 dB, SSIM = 0.6561
Epoch 9: PSNR = 22.89 dB, SSIM = 0.6878
Epoch 10: PSNR = 22.92 dB, SSIM = 0.6965
Epoch 11: PSNR = 23.27 dB, SSIM = 0.7062
Epoch 12: PSNR = 22.94 dB, SSIM = 0.7153
Epoch 13: PSNR = 24.37 dB, SSIM = 0.7277
Epoch 14: PSNR = 24.02 dB, SSIM = 0.7337
Epoch 15: PSNR = 25.08 dB, SSIM = 0.7515
Epoch 16: PSNR = 23.84 dB, SSIM = 0.7513
Epoch 17: PSNR = 24.64 dB, SSIM = 0.7567
Epoch 18: PSNR = 23.51 dB, SSIM = 0.7574
Epoch 19: PSNR = 25.61 dB, SSIM = 0.7714
Epoch 20: PSNR = 25.27 dB, SSIM = 0.7772
Epoch 21: PSNR = 25.85 dB, SSIM = 0.7794
Epoch 22: PSNR = 25.83 dB, SSIM = 0.7857
Epoch 23: PSNR = 25.33 dB, SSIM = 0.7830
Epoch 24: PSNR = 25.84 dB, SSIM = 0.7901
Epoch 25: PSNR = 26.25 dB, SSIM = 0.7935
Epoch 26: PSNR = 26.50 dB, SSIM = 0.7998
Epoch 27: PSNR = 25.95 dB, SSIM = 0.7902
Epoch 28: PSNR = 25.86 dB, SSIM = 0.8042
Epoch 29: PSNR = 26.76 dB, SSIM = 0.8090
Epoch 30: PSNR = 26.97 dB, SSIM = 0.8111
Epoch 31: PSNR = 27.12 dB, SSIM = 0.8154
Epoch 32: PSNR = 27.11 dB, SSIM = 0.8145
Epoch 33: PSNR = 27.44 dB, SSIM = 0.8214
Epoch 34: PSNR = 26.64 dB, SSIM = 0.8145
Epoch 35: PSNR = 26.94 dB, SSIM = 0.8207
Epoch 36: PSNR = 27.39 dB, SSIM = 0.8248
Epoch 37: PSNR = 26.82 dB, SSIM = 0.8252
Epoch 38: PSNR = 27.59 dB, SSIM = 0.8278
Epoch 39: PSNR = 27.54 dB, SSIM = 0.8289
Epoch 40: PSNR = 26.82 dB, SSIM = 0.8291
Epoch 41: PSNR = 27.20 dB, SSIM = 0.8315
Epoch 42: PSNR = 27.84 dB, SSIM = 0.8350
Epoch 43: PSNR = 26.70 dB, SSIM = 0.8223
Epoch 44: PSNR = 27.59 dB, SSIM = 0.8322
Epoch 45: PSNR = 28.01 dB, SSIM = 0.8406
Epoch 46: PSNR = 27.54 dB, SSIM = 0.8343
Epoch 47: PSNR = 28.17 dB, SSIM = 0.8445
Epoch 48: PSNR = 28.26 dB, SSIM = 0.8458
Epoch 49: PSNR = 27.87 dB, SSIM = 0.8423
Epoch 50: PSNR = 28.37 dB, SSIM = 0.8460
Epoch 51: PSNR = 28.58 dB, SSIM = 0.8484
Epoch 52: PSNR = 28.59 dB, SSIM = 0.8516
Epoch 53: PSNR = 28.42 dB, SSIM = 0.8498
Epoch 54: PSNR = 28.34 dB, SSIM = 0.8510
Epoch 55: PSNR = 28.62 dB, SSIM = 0.8505
Epoch 56: PSNR = 28.50 dB, SSIM = 0.8498
Epoch 57: PSNR = 28.83 dB, SSIM = 0.8561
Epoch 58: PSNR = 28.82 dB, SSIM = 0.8572
Epoch 59: PSNR = 28.49 dB, SSIM = 0.8546
Epoch 60: PSNR = 28.79 dB, SSIM = 0.8558
Epoch 61: PSNR = 28.95 dB, SSIM = 0.8584
Epoch 62: PSNR = 28.94 dB, SSIM = 0.8599
Epoch 63: PSNR = 28.87 dB, SSIM = 0.8607
Epoch 64: PSNR = 28.88 dB, SSIM = 0.8604
Epoch 65: PSNR = 28.88 dB, SSIM = 0.8598
Epoch 66: PSNR = 29.33 dB, SSIM = 0.8654
Epoch 67: PSNR = 29.23 dB, SSIM = 0.8668
Epoch 68: PSNR = 28.93 dB, SSIM = 0.8566
Epoch 69: PSNR = 29.20 dB, SSIM = 0.8661
Epoch 70: PSNR = 29.28 dB, SSIM = 0.8649
Epoch 71: PSNR = 29.37 dB, SSIM = 0.8653
Epoch 72: PSNR = 29.25 dB, SSIM = 0.8650
Epoch 73: PSNR = 29.10 dB, SSIM = 0.8673
Epoch 74: PSNR = 29.41 dB, SSIM = 0.8694
Epoch 75: PSNR = 28.86 dB, SSIM = 0.8644
Epoch 76: PSNR = 29.40 dB, SSIM = 0.8690
Epoch 77: PSNR = 29.39 dB, SSIM = 0.8703
Epoch 78: PSNR = 29.63 dB, SSIM = 0.8740
Epoch 79: PSNR = 29.36 dB, SSIM = 0.8738
Epoch 80: PSNR = 29.05 dB, SSIM = 0.8661
Epoch 81: PSNR = 29.80 dB, SSIM = 0.8748
Epoch 82: PSNR = 29.09 dB, SSIM = 0.8704
Epoch 83: PSNR = 29.52 dB, SSIM = 0.8738
Epoch 84: PSNR = 29.87 dB, SSIM = 0.8766
Epoch 85: PSNR = 29.97 dB, SSIM = 0.8790
Epoch 86: PSNR = 29.73 dB, SSIM = 0.8756
Epoch 87: PSNR = 29.83 dB, SSIM = 0.8782
Epoch 88: PSNR = 29.82 dB, SSIM = 0.8766
Epoch 89: PSNR = 29.62 dB, SSIM = 0.8748
Epoch 90: PSNR = 29.71 dB, SSIM = 0.8773
Epoch 91: PSNR = 29.77 dB, SSIM = 0.8791
Epoch 92: PSNR = 29.72 dB, SSIM = 0.8791
Epoch 93: PSNR = 29.94 dB, SSIM = 0.8795
Epoch 94: PSNR = 29.98 dB, SSIM = 0.8810
Epoch 95: PSNR = 29.85 dB, SSIM = 0.8811
Epoch 96: PSNR = 30.16 dB, SSIM = 0.8822
Epoch 97: PSNR = 29.95 dB, SSIM = 0.8796
Epoch 98: PSNR = 29.81 dB, SSIM = 0.8791
Epoch 99: PSNR = 30.22 dB, SSIM = 0.8837
Epoch 100: PSNR = 28.89 dB, SSIM = 0.8777
"""