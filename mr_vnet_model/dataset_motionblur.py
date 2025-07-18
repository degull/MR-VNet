import os
import csv
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class GoProDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.root_dir = root_dir
        csv_path = os.path.join(root_dir, csv_file)
        self.blur_paths = []
        self.sharp_paths = []

        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                blur_path = os.path.join(self.root_dir, row['blur_path']).replace('\\', '/')
                sharp_path = os.path.join(self.root_dir, row['sharp_path']).replace('\\', '/')
                self.blur_paths.append(blur_path)
                self.sharp_paths.append(sharp_path)

        self.transform = T.Compose([
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.blur_paths)

    def __getitem__(self, idx):
        blur_img = Image.open(self.blur_paths[idx]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[idx]).convert('RGB')

        blur_img = self.transform(blur_img)
        sharp_img = self.transform(sharp_img)

        return blur_img, sharp_img
