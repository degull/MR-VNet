import os
import csv
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms


def pil_loader(path):
    return Image.open(path).convert('RGB')


class KADID10KDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.samples = []
        self.transform = transform
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                dist_path = os.path.join(img_dir, row[0].strip())
                ref_path = os.path.join(img_dir, row[1].strip())
                self.samples.append((dist_path, ref_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dist_path, ref_path = self.samples[idx]
        dist = pil_loader(dist_path)
        ref = pil_loader(ref_path)
        
        # ✅ transform이 없으면 ToTensor()라도 해주자
        if self.transform:
            dist = self.transform(dist)
            ref = self.transform(ref)
        else:
            to_tensor = transforms.ToTensor()
            dist = to_tensor(dist)
            ref = to_tensor(ref)
        
        return dist, ref



class TID2013Dataset(Dataset):
    def __init__(self, csv_path, ref_dir, dist_dir, transform=None):
        self.samples = []
        self.transform = transform
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                dist_img = os.path.join(dist_dir, row[0].strip())
                ref_img = os.path.join(ref_dir, row[1].strip())
                self.samples.append((dist_img, ref_img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dist_path, ref_path = self.samples[idx]
        dist = pil_loader(dist_path)
        ref = pil_loader(ref_path)
        
        # ✅ transform이 없으면 ToTensor()라도 해주자
        if self.transform:
            dist = self.transform(dist)
            ref = self.transform(ref)
        else:
            to_tensor = transforms.ToTensor()
            dist = to_tensor(dist)
            ref = to_tensor(ref)
        
        return dist, ref



class CSIQDataset(Dataset):
    def __init__(self, txt_path, ref_dir, dist_root_dir, transform=None):
        self.samples = []
        self.transform = transform
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    ref_name, dist_subdir, dist_name = parts
                    ref_path = os.path.join(ref_dir, ref_name.strip())
                    dist_path = os.path.join(dist_root_dir, dist_subdir.strip(), dist_name.strip())
                    self.samples.append((dist_path, ref_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dist_path, ref_path = self.samples[idx]
        dist = pil_loader(dist_path)
        ref = pil_loader(ref_path)
        
        # ✅ transform이 없으면 ToTensor()라도 해주자
        if self.transform:
            dist = self.transform(dist)
            ref = self.transform(ref)
        else:
            to_tensor = transforms.ToTensor()
            dist = to_tensor(dist)
            ref = to_tensor(ref)
        
        return dist, ref



# ✅ KADID + TID2013 + CSIQ 통합
def get_all_datasets(transform=None):
    root = r'C:\Users\IIPL02\Desktop\MRVNet2D\dataset'

    kadid = KADID10KDataset(
        csv_path=os.path.join(root, 'KADID10K', 'kadid10k.csv'),
        img_dir=os.path.join(root, 'KADID10K', 'images'),
        transform=transform
    )

    tid = TID2013Dataset(
        csv_path=os.path.join(root, 'tid2013', 'mos.csv'),
        ref_dir=os.path.join(root, 'tid2013', 'reference_images'),
        dist_dir=os.path.join(root, 'tid2013', 'distorted_images'),
        transform=transform
    )

    csiq = CSIQDataset(
        txt_path=os.path.join(root, 'CSIQ', 'CSIQ.txt'),
        ref_dir=os.path.join(root, 'CSIQ', 'src_imgs'),
        dist_root_dir=os.path.join(root, 'CSIQ', 'dst_imgs'),
        transform=transform
    )

    return ConcatDataset([kadid, tid, csiq])
