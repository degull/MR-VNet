# reference x

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from mr_vnet_model.mrvnet_unet import MRVNetUNet

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\gopro\gopro_mrvnet_epoch95.pth"
DISTORTED_IMG_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I77_03_05.png"
SAVE_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\results\single_nogt"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

# ✅ 입력 이미지 불러오기
dist_img = load_image(DISTORTED_IMG_PATH)

# ✅ 모델 로딩
model = MRVNetUNet(in_channels=3, base_channels=32, rank=4).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    output = model(dist_img)

# ✅ 결과 저장
save_image(dist_img, f"{SAVE_DIR}/input.png")
save_image(output, f"{SAVE_DIR}/output.png")

print("\n📷 단일 이미지 복원 완료!")
print(f"→ 저장 위치: {SAVE_DIR}")
