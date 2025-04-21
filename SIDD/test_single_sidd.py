import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

from mr_vnet_model.mrvnet_unet import MRVNetUNet

# ✅ 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\sidd\sidd_mrvnet_epoch99.pth"
DISTORTED_IMG_PATH = r"C:\Users\IIPL02\Desktop\MRVNet2D\KADID10K\images\I77_03_05.png"
SAVE_DIR = r"C:\Users\IIPL02\Desktop\MRVNet2D\results\single_sidd"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ 전처리 정의
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_image(path):
    return transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)

# ✅ 입력 이미지 로드
dist_img = load_image(DISTORTED_IMG_PATH)

# ✅ 모델 로딩
model = MRVNetUNet(in_channels=3, base_channels=32, rank=4).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ✅ 복원 수행
with torch.no_grad():
    output = model(dist_img)

# ✅ input과 output을 좌우로 나란히 붙여 비교 이미지 생성
compare = torch.cat([dist_img, output], dim=3)  # dim=3: width 방향으로 연결
save_image(compare, f"{SAVE_DIR}/compare.png")

# ✅ 안내 메시지
print("\n✅ 복원 완료! 비교 이미지가 저장되었습니다.")
print(f"→ 비교 이미지 경로: {SAVE_DIR}/compare.png")
