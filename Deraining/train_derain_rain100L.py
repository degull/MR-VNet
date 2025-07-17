import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from mr_vnet_model.mrvnet_unet import MRVNetUNet
from mr_vnet_model.dataset_derain import RainDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = r'E:\MRVNet2D\dataset\rain100L'
save_dir = r'./checkpoints/rain100L'
os.makedirs(save_dir, exist_ok=True)

train_dataset = RainDataset(dataset_path, mode='train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

model = MRVNetUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for rain, norain in train_loader:
        rain, norain = rain.to(device), norain.to(device)
        restored = model(rain)
        loss = criterion(restored, norain)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Rain100L] Epoch [{epoch+1}/{num_epochs}]  Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
