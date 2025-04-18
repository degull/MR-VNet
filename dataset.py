# dataset.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageRestoreDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dist_name = self.data.iloc[idx]['dist_img']
        ref_name = self.data.iloc[idx]['ref_img']

        dist_path = os.path.join(self.img_dir, dist_name)
        ref_path = os.path.join(self.img_dir, ref_name)

        dist_img = Image.open(dist_path).convert("RGB")
        ref_img = Image.open(ref_path).convert("RGB")

        dist_img = self.transform(dist_img)
        ref_img = self.transform(ref_img)

        return dist_img, ref_img
