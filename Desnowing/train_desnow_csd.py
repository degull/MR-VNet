import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from mr_vnet_model.mrvnet_unet import MRVNetUNet
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


class CSDDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.input_dir = os.path.join(root_dir, mode.capitalize(), 'Snow')
        self.gt_dir = os.path.join(root_dir, mode.capitalize(), 'Gt')
        self.input_list = sorted(os.listdir(self.input_dir))
        self.gt_list = sorted(os.listdir(self.gt_dir))
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_list[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_list[idx])
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        input_img = self.transform(input_img)
        gt_img = self.transform(gt_img)
        return input_img, gt_img


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = r'E:\MRVNet2D\dataset\CSD'
    save_dir = r'./checkpoints/csd'
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = CSDDataset(dataset_path, mode='train')
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

        for idx, (snow, gt) in enumerate(train_loader):
            iter_start_time = time.time()

            snow, gt = snow.to(device), gt.to(device)
            restored = model(snow)
            loss = criterion(restored, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            snow_np = tensor_to_numpy(snow[0])
            gt_np = tensor_to_numpy(gt[0])
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

        print(f"[CSD] Epoch [{epoch+1}/{num_epochs}]  Avg Loss: {avg_loss:.4f}  Elapsed: {elapsed_str}")

        model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for snow, gt in train_loader:
                snow, gt = snow.to(device), gt.to(device)
                restored = model(snow)

                for i in range(snow.shape[0]):
                    gt_np = tensor_to_numpy(gt[i])
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
