# utils.py
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def tensor_to_img(tensor):
    """
    Converts torch tensor [B,C,H,W] or [C,H,W] to numpy [H,W,C] and clips to [0,1]
    """
    if tensor.dim() == 4:  # [1,C,H,W]
        tensor = tensor[0]
    np_img = tensor.detach().cpu().numpy()
    if np_img.shape[0] == 3:
        np_img = np.transpose(np_img, (1, 2, 0))  # [C,H,W] → [H,W,C]
    np_img = np.clip(np_img, 0, 1)
    return np_img

def calculate_psnr(output, target):
    output_np = tensor_to_img(output)
    target_np = tensor_to_img(target)
    return peak_signal_noise_ratio(target_np, output_np, data_range=1.0)

def calculate_ssim(output, target):
    output_np = tensor_to_img(output)
    target_np = tensor_to_img(target)
    return structural_similarity(target_np, output_np, data_range=1.0, channel_axis=-1)
