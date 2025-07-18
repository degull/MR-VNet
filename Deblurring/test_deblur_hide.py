import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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

    dataset_path = r'E:\MRVNet2D\dataset\HIDE'
    csv_file = 'hide_test_pairs.csv'   # 반드시 HIDE용 csv 준비되어 있어야 함
    save_dir = r'E:/MRVNet2D/results/HIDE'
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = GoProDataset(dataset_path, csv_file)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = MRVNetUNet().to(device)
    checkpoint_path = r'./checkpoints/gopro/epoch_99_ssim0.8837_psnr30.22.pth'  # GOPRO 학습 결과 사용
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    psnr_list, ssim_list = []

    total_iters = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for idx, (blur, sharp) in enumerate(test_loader):
            blur, sharp = blur.to(device), sharp.to(device)
            restored = model(blur)

            restored_clamp = torch.clamp(restored, 0, 1)
            save_path = os.path.join(save_dir, f'{idx+1:04d}.png')
            save_image(restored_clamp, save_path)

            gt_np = tensor_to_numpy(sharp[0])
            out_np = tensor_to_numpy(restored[0])

            psnr = compare_psnr(gt_np, out_np, data_range=255)
            ssim = compare_ssim(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            elapsed = time.time() - start_time
            eta = (total_iters - (idx + 1)) * (elapsed / (idx + 1))
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
            progress = (idx + 1) / total_iters * 100

            print(f"[Iter {idx+1}/{total_iters}]  PSNR: {psnr:.2f}  SSIM: {ssim:.4f}  "
                  f"Progress: {progress:.1f}%  ETA: {eta_str}  Elapsed: {elapsed_str}")

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"\n[HIDE] Test Average PSNR: {avg_psnr:.2f} dB  Average SSIM: {avg_ssim:.4f}")

    print("\n[INFO] All Iter PSNR / SSIM Results:")
    for i in range(len(psnr_list)):
        print(f"Iter {i+1}: PSNR = {psnr_list[i]:.2f} dB, SSIM = {ssim_list[i]:.4f}")


if __name__ == "__main__":
    main()
