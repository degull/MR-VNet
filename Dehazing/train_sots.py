# E:/MRVNet2D/Dehazing/train_sots_joint.py
import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from mr_vnet_model.mrvnet_unet import MRVNetUNet


# ---------------- Dataset ----------------
class SOTSDataset(Dataset):
    def __init__(self, csv_paths, root_dirs, transform=None):
        """
        csv_paths : [indoor_csv, outdoor_csv]
        root_dirs : [indoor_root, outdoor_root]
        """
        self.transform = transform
        self.pairs = []

        for csv_path, root_dir in zip(csv_paths, root_dirs):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                hazy_path = os.path.join(root_dir, row["cloud_image_path"])
                clear_path = os.path.join(root_dir, row["clear_image_path"])
                self.pairs.append((hazy_path, clear_path))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        hazy_path, clear_path = self.pairs[idx]
        hazy_img = Image.open(hazy_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")

        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)

        return hazy_img, clear_img


# ---------------- Utils ----------------
def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


# ---------------- Training ----------------
def train_sots_joint():
    # Dataset paths
    indoor_csv = r"E:\MRVNet2D\dataset\SOTS\metadata_indoor.csv"
    indoor_root = r"E:\MRVNet2D\dataset\SOTS"
    outdoor_csv = r"E:\MRVNet2D\dataset\SOTS\metadata_outdoor.csv"
    outdoor_root = r"E:\MRVNet2D\dataset\SOTS"

    SAVE_DIR = r"E:\MRVNet2D\checkpoints\sots_mrvnet"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Hyperparams
    BATCH_SIZE = 4
    NUM_EPOCHS = 100
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset (Indoor + Outdoor 합침)
    dataset = SOTSDataset(
        csv_paths=[indoor_csv, outdoor_csv],
        root_dirs=[indoor_root, outdoor_root],
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"[INFO] Total training pairs (Indoor + Outdoor): {len(dataset)}")

    # Model / Optimizer / Loss
    model = MRVNetUNet(in_channels=3).to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    psnr_per_epoch, ssim_per_epoch = [], []
    overall_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n[INFO] Joint SOTS → Starting Epoch {epoch}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0
        start_time = time.time()

        for hazy, clear in dataloader:
            hazy, clear = hazy.to(DEVICE), clear.to(DEVICE)
            restored = model(hazy)
            loss = criterion(restored, clear)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"[Joint SOTS] Epoch [{epoch}/{NUM_EPOCHS}]  Avg Loss: {avg_loss:.4f}  Time: {elapsed:.1f}s")

        # ---------- Evaluation ----------
        model.eval()
        psnr_list, ssim_list = [], []
        with torch.no_grad():
            for hazy, clear in dataloader:
                hazy, clear = hazy.to(DEVICE), clear.to(DEVICE)
                restored = model(hazy)
                for i in range(hazy.shape[0]):
                    gt_np = tensor_to_numpy(clear[i])
                    out_np = tensor_to_numpy(restored[i])
                    psnr = compute_psnr(gt_np, out_np, data_range=255)
                    ssim = compute_ssim(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        psnr_per_epoch.append(avg_psnr)
        ssim_per_epoch.append(avg_ssim)

        print(f"[Joint SOTS] Epoch {epoch} → Avg PSNR: {avg_psnr:.2f} dB  Avg SSIM: {avg_ssim:.4f}")

        # ---------- Save Checkpoint ----------
        filename = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, filename))

    print(f"\n[INFO] Joint SOTS Training Completed. All Epoch Results:")
    for i in range(NUM_EPOCHS):
        print(f"Epoch {i+1}: PSNR = {psnr_per_epoch[i]:.2f} dB, SSIM = {ssim_per_epoch[i]:.4f}")


if __name__ == "__main__":
    train_sots_joint()
