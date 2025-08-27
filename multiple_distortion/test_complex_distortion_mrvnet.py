# test_complex_distortion_mrvnet.py
import sys, os, glob, time
sys.path.append(r"E:/MRVNet2D/")

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
from tqdm import tqdm

from mr_vnet_model.mrvnet_unet import MRVNetUNet
from pipeline import apply_random_distortions   # 기존 multiple_distortion/pipeline.py 그대로 사용

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT   = r"E:\MRVNet2D\checkpoints\train_all_task\epoch_77_ssim0.8999_psnr31.55.pth"   # ✅ 체크포인트 경로 수정

# ----------------- 유틸 -----------------
def load_img(path, resize=256): 
    img = Image.open(path).convert("RGB")
    if resize:
        img = img.resize((resize, resize), Image.BICUBIC)
    return img

def tensor_to_numpy(t):
    arr = t.detach().cpu().numpy()
    arr = np.transpose(arr, (1,2,0))
    return np.clip(arr,0,1)

def load_model(ckpt_path):
    # MRVNetUNet은 out_channels 인자를 받지 않음 → 제거
    try:
        model = MRVNetUNet(in_channels=3).to(DEVICE)
    except TypeError:
        # 만약 in_channels도 필요 없는 경우
        model = MRVNetUNet().to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt if "state_dict" not in ckpt else ckpt["state_dict"])
    model.eval()
    return model

# ----------------- 실행 -----------------
if __name__ == "__main__":
    model = load_model(CKPT)
    tf = T.ToTensor()

    # GT pool
    gt_dirs = [
        r"E:/restormer+volterra/data/CSD/Test/Gt",
        r"E:/restormer+volterra/data/HIDE/GT",
        r"E:/restormer+volterra/data/rain100H/test/norain",
        r"E:/restormer+volterra/data/rain100L/test/norain",
        r"E:/restormer+volterra/data/SIDD/Data"
    ]

    gt_files = []
    for d in gt_dirs:
        if "SIDD" in d:
            gt_files += glob.glob(os.path.join(d, "**", "GT_SRGB_*.PNG"), recursive=True)
        else:
            gt_files += glob.glob(os.path.join(d, "*.*"))

    total_psnr, total_ssim = 0, 0
    count = 0
    start_time = time.time()

    for img_path in tqdm(gt_files, desc="Processing", ncols=100):
        clean = load_img(img_path, resize=256)
        distorted, info = apply_random_distortions(clean, Ndist=4, return_info=True)

        input_tensor = tf(distorted).unsqueeze(0).to(DEVICE)
        gt_tensor    = tf(clean).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            restored = model(input_tensor)

        psnr = compute_psnr(tensor_to_numpy(gt_tensor[0]), tensor_to_numpy(restored[0]), data_range=1.0)
        ssim = compute_ssim(tensor_to_numpy(gt_tensor[0]), tensor_to_numpy(restored[0]), channel_axis=-1, data_range=1.0)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

        # 진행률 및 ETA 출력
        elapsed = time.time() - start_time
        avg_time = elapsed / count
        eta = avg_time * (len(gt_files) - count)
        print(f"[{count}/{len(gt_files)}] {os.path.basename(img_path)} "
              f"| Distortions: {info} | PSNR={psnr:.2f}, SSIM={ssim:.4f} "
              f"| Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

    if count > 0:
        print("="*60)
        print(f"Processed {count} images")
        print(f"Final Average PSNR: {total_psnr/count:.2f}")
        print(f"Final Average SSIM: {total_ssim/count:.4f}")
        print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
