# E:/MRVNet2D/Dehazing/test_sots.py

import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import glob
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
    return transform(img).unsqueeze(0)  # 1xCxHxW


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


def pad_to_factor(x, factor=16):
    """입력을 factor 배수 크기로 reflect padding"""
    _, _, h, w = x.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    pad_h, pad_w = H - h, W - w
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, (h, w)


def crop_to_size(x, size):
    """패딩 전 크기로 crop"""
    h, w = size
    return x[:, :, :h, :w]


def evaluate_dataset(model, device, input_dir, target_dir, name):
    psnr_list, ssim_list = [], []

    if name == "SIDD":
        # ... (SIDD 부분은 그대로)
        pass

    else:
        # ✅ 하위폴더까지 input 탐색
        input_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True))
        input_files = [f for f in input_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # ✅ GT 전체 불러오기
        target_files = sorted(glob.glob(os.path.join(target_dir, "*.*")))
        target_files = [f for f in target_files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # ✅ GT를 딕셔너리 (basename → path) 로 인덱싱
        target_map = {os.path.splitext(os.path.basename(f))[0]: f for f in target_files}

        matched_pairs = []
        for inp_path in input_files:
            stem = os.path.splitext(os.path.basename(inp_path))[0]
            # input: 1fromGOPR1037.MP4.png → GT: 1fromGOPR1037.png
            stem = stem.split(".")[0]  # ".MP4" 같은 확장자 제거
            if stem in target_map:
                matched_pairs.append((inp_path, target_map[stem]))

        if len(matched_pairs) == 0:
            print(f"[ERROR] {name}: No matching pairs found!")
            return float("nan"), float("nan")

        print(f"[INFO] {name}: Matched {len(matched_pairs)} input/GT pairs")

        for inp_path, tgt_path in tqdm(matched_pairs):
            inp = load_image(inp_path).to(device)
            tgt = load_image(tgt_path).to(device)

            inp_pad, orig_size = pad_to_factor(inp, factor=16)
            with torch.no_grad():
                restored = model(inp_pad)
            restored = crop_to_size(restored, orig_size)

            out_np = tensor_to_numpy(restored[0])
            gt_np = tensor_to_numpy(tgt[0])

            psnr_list.append(compute_psnr(gt_np, out_np, data_range=1.0))
            ssim_list.append(compute_ssim(gt_np, out_np, channel_axis=-1, data_range=1.0))

    return np.mean(psnr_list), np.mean(ssim_list) if psnr_list else (float("nan"), float("nan"))



# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = r"E:\MRVNet2D\checkpoints\sots_mrvnet\epoch_91_ssim0.9616_psnr32.95.pth"

    DATASETS = {
        "HIDE": (r"E:/MRVNet2D/dataset/HIDE/test",   # input (recursive search)
                 r"E:/MRVNet2D/dataset/HIDE/GT"),    # GT
    }

    model = MRVNetUNet(in_channels=3).to(DEVICE)
    state_dict = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded checkpoint: {CKPT_PATH}")

    results = {}
    for name, (input_dir, target_dir) in DATASETS.items():
        print(f"\n[TEST] Evaluating {name} ...")
        psnr, ssim = evaluate_dataset(model, DEVICE, input_dir, target_dir, name)
        results[name] = (psnr, ssim)
        print(f"[RESULT] {name} → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    print("\n========== Final Results ==========")
    for name, (psnr, ssim) in results.items():
        print(f"{name:15s} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
