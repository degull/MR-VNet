# E:/MRVNet2D/Dehazing/test_sots.py

import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from tqdm import tqdm


# ---------------- Utils ----------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def pad_to_factor(x, factor=16):
    _, _, h, w = x.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    pad_h, pad_w = H - h, W - w
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (h, w)


def crop_to_size(x, size):
    h, w = size
    return x[:, :, :h, :w]


def evaluate_sots_from_csv(model, device, csv_path, dataset_root, name):
    df = pd.read_csv(csv_path)
    psnr_list, ssim_list = [], []

    print(f"[INFO] {name}: Found {len(df)} pairs from CSV")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        hazy_rel = row["cloud_image_path"]
        clear_rel = row["clear_image_path"]

        hazy_path = os.path.join(dataset_root, hazy_rel)
        clear_path = os.path.join(dataset_root, clear_rel)

        if not (os.path.exists(hazy_path) and os.path.exists(clear_path)):
            continue

        inp = load_image(hazy_path).to(device)
        tgt = load_image(clear_path).to(device)

        inp_pad, orig_size = pad_to_factor(inp, factor=16)
        with torch.no_grad():
            restored = model(inp_pad)
        restored = crop_to_size(restored, orig_size)

        out_np = tensor_to_numpy(restored[0])
        gt_np = tensor_to_numpy(tgt[0])

        psnr_list.append(compute_psnr(gt_np, out_np, data_range=1.0))
        ssim_list.append(compute_ssim(gt_np, out_np, channel_axis=-1, data_range=1.0))

    if len(psnr_list) == 0:
        return float("nan"), float("nan")

    return np.mean(psnr_list), np.mean(ssim_list)


# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = r"E:\MRVNet2D\checkpoints\sots_mrvnet\epoch_91_ssim0.9616_psnr32.95.pth"

    DATASETS = {
        "SOTS-indoor": {
            "csv": r"E:/MRVNet2D/dataset/SOTS/metadata_indoor.csv",
            "root": r"E:/MRVNet2D/dataset/SOTS"
        },
        "SOTS-outdoor": {
            "csv": r"E:/MRVNet2D/dataset/SOTS/metadata_outdoor.csv",
            "root": r"E:/MRVNet2D/dataset/SOTS"
        },
    }

    # 모델 로드
    model = MRVNetUNet(in_channels=3).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")

    results = {}
    for name, info in DATASETS.items():
        print(f"\n[TEST] Evaluating {name} ...")
        psnr, ssim = evaluate_sots_from_csv(model, DEVICE, info["csv"], info["root"], name)
        results[name] = (psnr, ssim)
        print(f"[RESULT] {name} → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # 요약 출력
    print("\n========== Final Results ==========")
    for name, (psnr, ssim) in results.items():
        print(f"{name:12s} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
