import os
import csv

def generate_gopro_csv(root_dir, split, output_csv):
    pairs = []
    split_dir = os.path.join(root_dir, split)
    for seq_name in os.listdir(split_dir):
        blur_dir = os.path.join(split_dir, seq_name, 'blur')
        sharp_dir = os.path.join(split_dir, seq_name, 'sharp')
        if not os.path.exists(blur_dir) or not os.path.exists(sharp_dir):
            continue
        for img_name in sorted(os.listdir(blur_dir)):
            dist_path = os.path.join(blur_dir, img_name).replace('\\', '/')
            ref_path = os.path.join(sharp_dir, img_name).replace('\\', '/')
            if os.path.exists(dist_path) and os.path.exists(ref_path):
                pairs.append([dist_path, ref_path])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dist_img', 'ref_img'])
        writer.writerows(pairs)

# 예시 실행
# 출력 경로 설정을 명확하게!
generate_gopro_csv(
    root_dir=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\GOPRO_Large",
    split='train',
    output_csv=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\gopro_train_pairs.csv"
)

generate_gopro_csv(
    root_dir=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\GOPRO_Large",
    split='test',
    output_csv=r"C:\Users\IIPL02\Desktop\MRVNet2D\dataset\gopro_test_pairs.csv"
)
