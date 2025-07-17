import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from mr_vnet_model.dataset_derain import RainDataset
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def save_numpy_to_png(np_img, path):
    img_pil = Image.fromarray(np_img)
    img_pil.save(path)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = r'E:\MRVNet2D\dataset\rain100L'
    model_path = r'./checkpoints/rain100L/epoch_100_ssim0.9400_psnr33.00.pth'  # 네가 저장한 모델 경로
    save_dir = r'E:/MRVNet2D/results/Rain100L'
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = RainDataset(dataset_path, mode='test')
    print(f"[INFO] Found {len(test_dataset)} test samples")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = MRVNetUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    psnr_list, ssim_list = []

    total_iters = len(test_loader)
    overall_start_time = time.time()

    with torch.no_grad():
        for idx, (rain, norain) in enumerate(test_loader):
            iter_start_time = time.time()

            rain, norain = rain.to(device), norain.to(device)
            restored = model(rain)

            rain_np = tensor_to_numpy(rain[0])
            gt_np = tensor_to_numpy(norain[0])
            out_np = tensor_to_numpy(restored[0])

            psnr = compare_psnr(gt_np, out_np, data_range=255)
            ssim = compare_ssim(gt_np, out_np, channel_axis=-1, data_range=255, win_size=7)

            psnr_list.append(psnr)
            ssim_list.append(ssim)

            # 복원 이미지 저장
            save_path = os.path.join(save_dir, f"{idx+1:03d}_restored.png")
            save_numpy_to_png(out_np, save_path)

            iter_time = time.time() - iter_start_time
            remain_iters = total_iters - (idx + 1)
            remain_time = remain_iters * iter_time
            remain_time_str = time.strftime('%H:%M:%S', time.gmtime(remain_time))

            progress = (idx + 1) / total_iters * 100
            elapsed_training_time = time.time() - overall_start_time
            elapsed_training_time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_training_time))

            print(f"[Iter {idx+1}/{total_iters}]  "
                  f"PSNR: {psnr:.2f}  SSIM: {ssim:.4f}  "
                  f"Progress: {progress:.1f}%  ETA: {remain_time_str}  "
                  f"Elapsed: {elapsed_training_time_str}")

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    # 개별 PSNR / SSIM 전부 출력
    print("\n[INFO] All Iter PSNR / SSIM Results:")
    for i in range(total_iters):
        print(f"Iter {i+1}: PSNR = {psnr_list[i]:.2f} dB, SSIM = {ssim_list[i]:.4f}")

    # 평균 출력
    print(f"\n[Rain100L] Test Average PSNR: {avg_psnr:.2f} dB  Average SSIM: {avg_ssim:.4f}")


if __name__ == "__main__":
    main()
