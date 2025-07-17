import os
import torch
from torch.utils.data import DataLoader
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from mr_vnet_model.dataset_derain import RainDataset
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = r'E:\MRVNet2D\dataset\rain100L'
save_dir = r'./results/rain100L'
os.makedirs(save_dir, exist_ok=True)

test_dataset = RainDataset(dataset_path, mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = MRVNetUNet().to(device)
model.load_state_dict(torch.load('./checkpoints/rain100L/epoch_200.pth', map_location=device))
model.eval()

with torch.no_grad():
    for idx, (rain, _) in enumerate(test_loader):
        rain = rain.to(device)
        restored = model(rain)
        save_path = os.path.join(save_dir, f'{idx:04d}.png')
        save_image(restored.clamp(0, 1), save_path)

print(f"[Rain100L] Results saved to {save_dir}")
