import os
import csv

def generate_sidd_csv(data_dir, output_csv):
    """
    SIDD_Small_sRGB_Only 폴더 내부에서 NOISY_SRGB_010.PNG와 GT_SRGB_010.PNG 쌍을 찾아
    CSV로 저장하는 함수입니다.
    
    Args:
        data_dir (str): 폴더 위치 예시: C:/.../SIDD_Small_sRGB_Only/Data
        output_csv (str): 저장할 CSV 파일 경로
    """
    pairs = []

    # Data 폴더 내부 순회
    for folder in os.listdir(data_dir):
        subdir = os.path.join(data_dir, folder)
        if not os.path.isdir(subdir):
            continue

        noisy_path = os.path.join(subdir, "NOISY_SRGB_010.PNG")
        gt_path = os.path.join(subdir, "GT_SRGB_010.PNG")

        # 두 파일이 다 존재하면 추가
        if os.path.exists(noisy_path) and os.path.exists(gt_path):
            pairs.append([
                noisy_path.replace("\\", "/"),  # 경로 통일 (Windows → Linux 호환)
                gt_path.replace("\\", "/")
            ])

    # CSV 파일로 저장
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dist_img', 'ref_img'])  # 헤더
        writer.writerows(pairs)

# ✅ 실행 예시
generate_sidd_csv(
    data_dir=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\SIDD_Small_sRGB_Only\Data",
    output_csv=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\sidd_pairs.csv"
)
