import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from mr_vnet_model.mrvnet_unet import MRVNetUNet


# ✅ 설정
checkpoint_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\checkpoints\re_mrvnet_epoch98.pth"
input_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\000627.png"  # 복원할 이미지 경로
output_image_path = r"C:\Users\IIPL02\Desktop\MRVNet2D\single_results\result.png"
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 불러오기
model = MRVNetUNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 입력 이미지 로드
input_image = Image.open(input_image_path).convert("RGB")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# ✅ 복원 수행
with torch.no_grad():
    output_tensor = model(input_tensor).squeeze(0).clamp(0, 1).cpu()

# ✅ 시각화 및 저장
def visualize_and_save(input_img, output_img, save_path):
    import torchvision
    from torchvision.utils import make_grid
    import numpy as np

    grid = make_grid([input_img, output_img], nrow=2)
    np_img = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(8, 4))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title('Blurred / Restored')
    plt.savefig(save_path)
    plt.close()

visualize_and_save(input_tensor.squeeze(0).cpu(), output_tensor, output_image_path)
print(f"✅ 복원 결과 저장 완료: {output_image_path}")
