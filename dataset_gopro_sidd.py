import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PairedImageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_path = self.data.iloc[idx]['dist_img']
        ref_path = self.data.iloc[idx]['ref_img']

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        return self.transform(dist_img), self.transform(ref_img)
