import sys
sys.path.append(r"E:/MRVNet2D/")

import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
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
    def __init__(self, root_dir, mode='test'):
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
    save_dir = r'E:/MRVNet2D/results/CSD'
    os.makedirs(save_dir, exist_ok=True)

    test_dataset = CSDDataset(dataset_path, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = MRVNetUNet().to(device)
    checkpoint_path = r'./checkpoints/csd/epoch_100_ssim0.8xxx_psnr2x.xx.pth'  # 학습 결과 파일 지정
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    psnr_list, ssim_list = []

    total_iters = len(test_loader)
    start_time = time.time()

    with torch.no_grad():
        for idx, (snow, gt) in enumerate(test_loader):
            snow, gt = snow.to(device), gt.to(device)
            restored = model(snow)

            restored_clamp = torch.clamp(restored, 0, 1)
            save_path = os.path.join(save_dir, f'{idx+1:04d}.png')
            save_image(restored_clamp, save_path)

            gt_np = tensor_to_numpy(gt[0])
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

    print(f"\n[CSD] Test Average PSNR: {avg_psnr:.2f} dB  Average SSIM: {avg_ssim:.4f}")

    print("\n[INFO] All Iter PSNR / SSIM Results:")
    for i in range(len(psnr_list)):
        print(f"Iter {i+1}: PSNR = {psnr_list[i]:.2f} dB, SSIM = {ssim_list[i]:.4f}")


if __name__ == "__main__":
    main()
