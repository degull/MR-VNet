# E:/MRVNet2D/test_all_tasks_mrvnet.py
# - 모델: MRVNetUNet (E:\MRVNet2D\mr_vnet_model)
# - 테스트: Rain100H/L (test), GoPro(test csv), HIDE, SIDD(test csv),
#           CSD(Test), Classic5, LIVE1, SOTS(indoor/outdoor), (옵션) BSDS500 val/test
#           + KADID separate (gaussian blur, impulse noise, white noise)
# - 메트릭: PSNR/SSIM
# - 옵션: 복원 결과 이미지 저장(save_images=True)

import os, sys, glob, time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ───────────────────────────────────────────────────────────
# 경로/모델 import
# ───────────────────────────────────────────────────────────
sys.path.append(r"E:/MRVNet2D/")
from mr_vnet_model.mrvnet_unet import MRVNetUNet  # 네 모델

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────
# 설정
# ───────────────────────────────────────────────────────────
CKPT_DIR = r"E:/MRVNet2D/checkpoints/train_all_task"  # 폴더 경로
CKPT_PATH = r"E:/MRVNet2D/checkpoints/train_all_task/epoch_75_ssim0.9005_psnr31.70.pth"

BATCH_SIZE = 1
NUM_WORKERS = 2
SAVE_IMAGES = False  # True로 하면 복원이미지 저장
SAVE_ROOT = r"E:/MRVNet2D/test_outputs"  # SAVE_IMAGES=True일 때 저장 폴더

# 테스트 시 해상도(학습 최종과 맞춤: 256)
def get_test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

# ───────────────────────────────────────────────────────────
# 데이터셋 유틸
# ───────────────────────────────────────────────────────────
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _find_gt_by_stem(target_dir, stem):
    """target_dir에서 stem과 동일한 파일(확장자 무관)을 찾는다."""
    for ext in VALID_EXT:
        cand = os.path.join(target_dir, stem + ext)
        if os.path.exists(cand):
            return cand
    return None

class PairedFolderDataset(Dataset):
    """
    입력/정답 폴더 쌍을 구성하는 Dataset.
    - 1차: 입력파일 stem과 완전 동일한 stem의 GT 매칭
    - 2차(SOTS 대응): stem.split('_')[0]로 축약해 GT 매칭 (예: 1400_1.png -> 1400.png)
    """
    def __init__(self, input_dir, target_dir, transform):
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXT)])
        self.input_paths, self.target_paths = [], []
        miss = 0

        for fname in files:
            inp = os.path.join(input_dir, fname)
            stem = os.path.splitext(fname)[0]

            gt = _find_gt_by_stem(target_dir, stem)
            if gt is None and "_" in stem:
                prefix = stem.split("_")[0]
                gt = _find_gt_by_stem(target_dir, prefix)

            if gt is not None:
                self.input_paths.append(inp)
                self.target_paths.append(gt)
            else:
                miss += 1

        if miss > 0:
            print(f"[PairedFolderDataset] {os.path.basename(input_dir)}: {miss} file(s) had no GT match.")

        self.transform = transform

    def __len__(self): 
        return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt), os.path.basename(self.input_paths[idx])

class PairedCSVDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        if df.shape[1] < 2:
            raise ValueError(f"CSV must have >=2 columns (input_path, target_path): {csv_path}")
        self.pairs = df.values.tolist()
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx][0], self.pairs[idx][1]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt), os.path.basename(inp_path)

class PairedListDataset(Dataset):
    """
    공백/쉼표/탭으로 구분된 txt 목록에서 (input, target) 쌍을 읽는다.
    - 라인 예시:
      E:/.../blur/level_1/abcd.png  E:/.../gt/abcd.png
      E:/.../blur/level_1/abcd.png, E:/.../gt/abcd.png
    - 상대경로면 txt 파일이 있는 디렉토리를 기준으로 해석한다.
    - 주석(#) 라인은 무시.
    """
    def __init__(self, list_path, transform):
        self.transform = transform
        self.base_dir = os.path.dirname(list_path)
        self.pairs = []
        miss = 0

        with open(list_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

        for ln in lines:
            # 쉼표 우선 split, 없으면 공백/탭 split
            parts = [p.strip().strip('"').strip("'") for p in (ln.split(",") if ("," in ln) else ln.split())]
            if len(parts) < 2:
                miss += 1
                continue

            inp_path, tgt_path = parts[0], parts[1]
            if not os.path.isabs(inp_path):
                inp_path = os.path.normpath(os.path.join(self.base_dir, inp_path))
            if not os.path.isabs(tgt_path):
                tgt_path = os.path.normpath(os.path.join(self.base_dir, tgt_path))

            if not (os.path.exists(inp_path) and os.path.exists(tgt_path)):
                miss += 1
                continue

            self.pairs.append((inp_path, tgt_path))

        if miss > 0:
            print(f"[PairedListDataset] {os.path.basename(list_path)}: skipped {miss} invalid/missing pair(s).")
        if len(self.pairs) == 0:
            raise FileNotFoundError(f"No valid pairs found in {list_path}")

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt), os.path.basename(inp_path)

class Classic5Dataset(Dataset):
    # classic5/
    #   refimgs/*.bmp
    #   gray/qf_*/ *.jpg
    def __init__(self, root, transform):
        ref_dir = os.path.join(root, "refimgs")
        dist_glob = glob.glob(os.path.join(root, "gray", "qf_*", "*.*"))
        ref_map = {}
        for p in glob.glob(os.path.join(ref_dir, "*.*")):
            ref_map[os.path.splitext(os.path.basename(p))[0]] = p
        pairs = []
        for dp in sorted(dist_glob):
            stem = os.path.splitext(os.path.basename(dp))[0]
            gt = ref_map.get(stem, None)
            if gt: pairs.append((dp, gt))
        self.pairs = pairs
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt), os.path.basename(inp_path)

class LIVE1Dataset(Dataset):
    # LIVE1/
    #   refimgs/*,  jpeg/qf_*/ *.jpg  (혹은 구조 변형)
    def __init__(self, root, transform):
        ref_dir = None
        for cand in ["refimgs", "reference", "gt", "GT", "ref"]:
            d = os.path.join(root, cand)
            if os.path.isdir(d):
                ref_dir = d; break
        if ref_dir is None:
            raise FileNotFoundError(f"LIVE1 reference dir not found under: {root}")

        dist_glob = []
        for sub in ["jpeg", "JPEG", "compressed", "images"]:
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                dist_glob += glob.glob(os.path.join(d, "qf_*", "*.*"))
        if not dist_glob:
            dist_glob = glob.glob(os.path.join(root, "qf_*", "*.*"))

        ref_map = {}
        for p in glob.glob(os.path.join(ref_dir, "*.*")):
            ref_map[os.path.splitext(os.path.basename(p))[0]] = p

        pairs = []
        for dp in sorted(dist_glob):
            stem = os.path.splitext(os.path.basename(dp))[0]
            gt = ref_map.get(stem, None)
            if gt: pairs.append((dp, gt))
        self.pairs = pairs
        self.transform = transform

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt), os.path.basename(inp_path)

# ───────────────────────────────────────────────────────────
# 경로(네 환경)
# ───────────────────────────────────────────────────────────
PATHS = {
    "rain100h_test_rain": r"E:/restormer+volterra/data/rain100H/test/rain",
    "rain100h_test_gt"  : r"E:/restormer+volterra/data/rain100H/test/norain",
    "rain100l_test_rain": r"E:/restormer+volterra/data/rain100L/test/rain",
    "rain100l_test_gt"  : r"E:/restormer+volterra/data/rain100L/test/norain",
    "gopro_test_csv"    : r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv",
    "hide_test_blur"    : r"E:/restormer+volterra/data/HIDE/test",
    "hide_test_gt"      : r"E:/restormer+volterra/data/HIDE/GT",
    "sidd_test_csv"     : r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv",
    "csd_test_snow"     : r"E:/restormer+volterra/data/CSD/Test/Snow",
    "csd_test_gt"       : r"E:/restormer+volterra/data/CSD/Test/Gt",

    # ✅ SOTS는 MRVNet2D 경로
    "SOTS-indoor"       : r"E:/MRVNet2D/dataset/SOTS/indoor/hazy",
    "SOTS-indoor_gt"    : r"E:/MRVNet2D/dataset/SOTS/indoor/clear",
    "SOTS-outdoor"      : r"E:/MRVNet2D/dataset/SOTS/outdoor/hazy",
    "SOTS-outdoor_gt"   : r"E:/MRVNet2D/dataset/SOTS/outdoor/clear",

    # ✅ KADID separate (TXT pair lists)
    "kadid_gaussian_txt": r"E:/MRVNet2D/dataset/kadid_seperate/gaussian/pairs_gaussian.txt",
    "kadid_impulse_txt" : r"E:/MRVNet2D/dataset/kadid_seperate/impulse noise/pairs_impulse.txt",
    "kadid_white_txt"   : r"E:/MRVNet2D/dataset/kadid_seperate/white noise/pairs_white.txt",
}

# ───────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────
def latest_checkpoint(ckpt_dir):
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")), key=os.path.getmtime)
    return cands[-1] if cands else ""

def tensor_to_uint8(img_tensor):
    # [C,H,W] tensor in [0,1] -> uint8 HWC
    x = img_tensor.detach().cpu().clamp(0,1).numpy()
    x = np.transpose(x, (1,2,0))
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

@torch.no_grad()
def evaluate_dataset(model, loader, save_dir=None):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0

    for inp, tgt, name in tqdm(loader, desc="  - infer"):
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        out = model(inp).clamp(0, 1)

        # metrics
        outs = out.detach().cpu().numpy()
        gts  = tgt.detach().cpu().numpy()
        for o, g in zip(outs, gts):
            o = np.transpose(o, (1, 2, 0))
            g = np.transpose(g, (1, 2, 0))
            total_psnr += compute_psnr(g, o, data_range=1.0)
            total_ssim += compute_ssim(g, o, data_range=1.0, channel_axis=-1)
            n += 1

        # save result
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for i in range(out.shape[0]):
                out_img = tensor_to_uint8(out[i])
                Image.fromarray(out_img).save(os.path.join(save_dir, f"{name[i]}"))

    psnr = total_psnr / max(1, n)
    ssim = total_ssim / max(1, n)
    return psnr, ssim, n

def build_test_suites(transform):
    suites = {}

    def add_if_exists(key, ds):
        try:
            if len(ds) > 0:
                suites[key] = ds
        except Exception as e:
            print(f"[skip] {key}: {e}")

    print("🧪 Collecting test datasets ...")

    # 기존 세트
    if os.path.isdir(PATHS["rain100h_test_rain"]) and os.path.isdir(PATHS["rain100h_test_gt"]):
        add_if_exists("Rain100H(test)",
                      PairedFolderDataset(PATHS["rain100h_test_rain"], PATHS["rain100h_test_gt"], transform))

    if os.path.isdir(PATHS["rain100l_test_rain"]) and os.path.isdir(PATHS["rain100l_test_gt"]):
        add_if_exists("Rain100L(test)",
                      PairedFolderDataset(PATHS["rain100l_test_rain"], PATHS["rain100l_test_gt"], transform))

    if os.path.isfile(PATHS["gopro_test_csv"]):
        add_if_exists("GoPro(test)",
                      PairedCSVDataset(PATHS["gopro_test_csv"], transform))

    if os.path.isdir(PATHS["hide_test_blur"]) and os.path.isdir(PATHS["hide_test_gt"]):
        add_if_exists("HIDE",
                      PairedFolderDataset(PATHS["hide_test_blur"], PATHS["hide_test_gt"], transform))

    if os.path.isfile(PATHS["sidd_test_csv"]):
        add_if_exists("SIDD(test)",
                      PairedCSVDataset(PATHS["sidd_test_csv"], transform))

    if os.path.isdir(PATHS["csd_test_snow"]) and os.path.isdir(PATHS["csd_test_gt"]):
        add_if_exists("CSD(Test)",
                      PairedFolderDataset(PATHS["csd_test_snow"], PATHS["csd_test_gt"], transform))

    # ✅ SOTS(indoor/outdoor)
    if os.path.isdir(PATHS["SOTS-indoor"]) and os.path.isdir(PATHS["SOTS-indoor_gt"]):
        add_if_exists("SOTS-Indoor",
                      PairedFolderDataset(PATHS["SOTS-indoor"], PATHS["SOTS-indoor_gt"], transform))

    if os.path.isdir(PATHS["SOTS-outdoor"]) and os.path.isdir(PATHS["SOTS-outdoor_gt"]):
        add_if_exists("SOTS-Outdoor",
                      PairedFolderDataset(PATHS["SOTS-outdoor"], PATHS["SOTS-outdoor_gt"], transform))

    # ✅ KADID separate (TXT)
    if os.path.isfile(PATHS["kadid_gaussian_txt"]):
        add_if_exists("KADID-GaussianBlur",
                      PairedListDataset(PATHS["kadid_gaussian_txt"], transform))

    if os.path.isfile(PATHS["kadid_impulse_txt"]):
        add_if_exists("KADID-ImpulseNoise",
                      PairedListDataset(PATHS["kadid_impulse_txt"], transform))

    if os.path.isfile(PATHS["kadid_white_txt"]):
        add_if_exists("KADID-WhiteNoise",
                      PairedListDataset(PATHS["kadid_white_txt"], transform))

    return suites


def main():
    # checkpoint 결정
    ckpt = CKPT_PATH if CKPT_PATH and os.path.exists(CKPT_PATH) else latest_checkpoint(CKPT_DIR)
    if not ckpt:
        raise FileNotFoundError(f"No checkpoint found. Set CKPT_PATH or put .pth under {CKPT_DIR}")
    print(f"✅ Using checkpoint: {ckpt}")

    # 모델 로드 (PyTorch 최신 권장: weights_only=True, 구버전 호환)
    model = MRVNetUNet().to(DEVICE)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)

    # 테스트 데이터셋 준비
    transform = get_test_transform()
    suites = build_test_suites(transform)

    # 평가 루프
    print("\n🔎 Evaluating ...\n")
    for name, ds in suites.items():
        out_dir = None
        if SAVE_IMAGES:
            out_dir = os.path.join(SAVE_ROOT, os.path.basename(ckpt).replace(".pth",""), name.replace("(","_").replace(")",""))
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        psnr, ssim, n = evaluate_dataset(model, loader, save_dir=out_dir)
        print(f"[{name:16s}] N={n:5d} | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}")
    print("")

if __name__ == "__main__":
    main()
