# E:/MRVNet2D/Dehazing/test_kadid.py

import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet


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


def evaluate_from_pairs(model, device, pairs_txt, root_dir, name):
    with open(pairs_txt, "r") as f:
        lines = f.readlines()

    psnr_list, ssim_list = [], []
    print(f"[INFO] {name}: Found {len(lines)} pairs")

    for line in tqdm(lines):
        inp_rel, tgt_rel = line.strip().split()
        inp_path = os.path.join(root_dir, inp_rel)
        tgt_path = os.path.join(root_dir, tgt_rel)

        if not (os.path.exists(inp_path) and os.path.exists(tgt_path)):
            continue

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

    if len(psnr_list) == 0:
        return float("nan"), float("nan")

    return np.mean(psnr_list), np.mean(ssim_list)


# ---------------- Main ----------------
def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT_PATH = r"E:\MRVNet2D\checkpoints\sots_mrvnet\epoch_89_ssim0.9613_psnr31.80.pth"

    DATASETS = {
        "KADID-gaussian": {
            "root": r"E:/MRVNet2D/dataset/kadid_seperate/gaussian",
            "pairs": r"E:/MRVNet2D/dataset/kadid_seperate/gaussian/pairs_gaussian.txt",
        },
        "KADID-impulse": {
            "root": r"E:/MRVNet2D/dataset/kadid_seperate/impulse noise",
            "pairs": r"E:/MRVNet2D/dataset/kadid_seperate/impulse noise/pairs_impulse.txt",
        },
        "KADID-white": {
            "root": r"E:/MRVNet2D/dataset/kadid_seperate/white noise",
            "pairs": r"E:/MRVNet2D/dataset/kadid_seperate/white noise/pairs_white.txt",
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
        psnr, ssim = evaluate_from_pairs(model, DEVICE, info["pairs"], info["root"], name)
        results[name] = (psnr, ssim)
        print(f"[RESULT] {name} → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

    # 요약 출력
    print("\n========== Final Results ==========")
    for name, (psnr, ssim) in results.items():
        print(f"{name:14s} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")


if __name__ == "__main__":
    main()
