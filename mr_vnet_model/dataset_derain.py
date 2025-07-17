import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class RainDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        assert mode in ['train', 'test']
        self.rain_dir = os.path.join(root_dir, mode, 'rain')
        self.norain_dir = os.path.join(root_dir, mode, 'norain')

        self.rain_images = sorted(os.listdir(self.rain_dir))
        self.norain_images = sorted(os.listdir(self.norain_dir))

        # 모든 이미지를 256x256으로 고정
        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rain_images)

    def __getitem__(self, idx):
        rain_path = os.path.join(self.rain_dir, self.rain_images[idx])
        norain_path = os.path.join(self.norain_dir, self.norain_images[idx])

        #print(f"Loading rain: {rain_path}, norain: {norain_path}")

        rain_img = Image.open(rain_path).convert('RGB')
        norain_img = Image.open(norain_path).convert('RGB')

        rain_img = self.transform(rain_img)
        norain_img = self.transform(norain_img)

        return rain_img, norain_img

