# E:/MRVNet2D/train_and_test_all_balanced_mrvnet.py
# - 모델: MRVNetUNet (E:\MRVNet2D\mr_vnet_model)
# - 학습: Deraining + Deblurring + Denoising + Desnowing + JPEG (각 160장 균형)
# - 테스트: 학습에 사용하지 않은 split/데이터셋 (존재하는 항목만 자동 평가)
# - 로그: ETA, 경과 시간, 진행률, Epoch별 PSNR/SSIM
# - 체크포인트 파일명: epoch_{e}_ssim{:.4f}_psnr{:.2f}.pth

import os, sys, time, random, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ─────────────────────────────────────────────────────────────────────
# 경로/모델 import (MRVNet만 사용)
# ─────────────────────────────────────────────────────────────────────
sys.path.append(r"E:/MRVNet2D/")
from mr_vnet_model.mrvnet_unet import MRVNetUNet  # ← 네 모델

# ─────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1234
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 저장 경로 (요청 경로)
SAVE_DIR = r"E:/MRVNet2D/checkpoints/train_all_task"
os.makedirs(SAVE_DIR, exist_ok=True)

# 재시작 옵션(필요 없으면 빈 문자열)
RESUME_PATH = ""  # 예: r"E:/MRVNet2D/checkpoints/train_all_task/epoch_46_ssim0.9119_psnr32.82.pth"
START_EPOCH = 1   # RESUME 쓰면 적절히 수정
EPOCHS = 100
BATCH_SIZE = 2
LR = 2e-4
MAX_SAMPLES = 160       # 각 태스크 균형 샘플 개수
EVAL_EVERY = 0          # 0이면 마지막만 평가, N이면 N에폭마다 평가

# 리사이즈 스케줄
resize_schedule = {0: 128, 30: 192, 60: 256}
def get_transform(epoch):
    size = max(v for k, v in resize_schedule.items() if epoch >= k)
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

# ─────────────────────────────────────────────────────────────────────
# 데이터셋 유틸
# ─────────────────────────────────────────────────────────────────────
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class PairedFolderDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform):
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(VALID_EXT)])
        self.input_paths, self.target_paths = [], []
        for fname in files:
            inp_path = os.path.join(input_dir, fname)
            stem = os.path.splitext(fname)[0]
            gt_path = None
            for ext in VALID_EXT:
                cand = os.path.join(target_dir, stem + ext)
                if os.path.exists(cand):
                    gt_path = cand; break
            if gt_path is None:
                print(f"[⚠️] GT not found for {fname}")
                continue
            self.input_paths.append(inp_path)
            self.target_paths.append(gt_path)
        assert len(self.input_paths) == len(self.target_paths), "Input/target mismatch"
        self.transform = transform

    def __len__(self): return len(self.input_paths)

    def __getitem__(self, idx):
        inp = Image.open(self.input_paths[idx]).convert("RGB")
        tgt = Image.open(self.target_paths[idx]).convert("RGB")
        return self.transform(inp), self.transform(tgt)

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
        return self.transform(inp), self.transform(tgt)

def subset_dataset(dataset, max_samples, seed=SEED):
    total = len(dataset)
    if total <= max_samples: return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(total), max_samples)
    return Subset(dataset, indices)

# Classic5 / LIVE1 전용 (파일명 매칭 기반)
class Classic5Dataset(Dataset):
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
            if gt is None: continue
            pairs.append((dp, gt))
        self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)

class LIVE1Dataset(Dataset):
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
            if gt is None: continue
            pairs.append((dp, gt))
        self.pairs = pairs
        self.transform = transform
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        inp_path, tgt_path = self.pairs[idx]
        inp = Image.open(inp_path).convert("RGB")
        tgt = Image.open(tgt_path).convert("RGB")
        return self.transform(inp), self.transform(tgt)

# ─────────────────────────────────────────────────────────────────────
# 경로(네 환경 기준)
# ─────────────────────────────────────────────────────────────────────
PATHS = {
    # Train (balanced 160ea)
    "rain100h_train_rain": r"E:/restormer+volterra/data/rain100H/train/rain",
    "rain100h_train_gt"  : r"E:/restormer+volterra/data/rain100H/train/norain",
    "rain100l_train_rain": r"E:/restormer+volterra/data/rain100L/train/rain",
    "rain100l_train_gt"  : r"E:/restormer+volterra/data/rain100L/train/norain",
    "gopro_train_csv"    : r"E:/restormer+volterra/data/GOPRO_Large/gopro_train_pairs.csv",
    "sidd_train_csv"     : r"E:/restormer+volterra/data/SIDD/sidd_pairs.csv",
    "csd_train_snow"     : r"E:/restormer+volterra/data/CSD/Train/Snow",
    "csd_train_gt"       : r"E:/restormer+volterra/data/CSD/Train/Gt",
    "bsds_train_img"     : r"E:/restormer+volterra/data/BSDS500/images/train",
    "bsds_train_gt"      : r"E:/restormer+volterra/data/BSDS500/ground_truth/train",

    # Test (unseen wrt training)
    "rain100h_test_rain" : r"E:/restormer+volterra/data/rain100H/test/rain",
    "rain100h_test_gt"   : r"E:/restormer+volterra/data/rain100H/test/norain",
    "rain100l_test_rain" : r"E:/restormer+volterra/data/rain100L/test/rain",
    "rain100l_test_gt"   : r"E:/restormer+volterra/data/rain100L/test/norain",
    "gopro_test_csv"     : r"E:/restormer+volterra/data/GOPRO_Large/gopro_test_pairs.csv",
    "hide_test_blur"     : r"E:/restormer+volterra/data/HIDE/test",
    "hide_test_gt"       : r"E:/restormer+volterra/data/HIDE/GT",
    "sidd_test_csv"      : r"E:/restormer+volterra/data/SIDD/sidd_test_pairs.csv",
    "csd_test_snow"      : r"E:/restormer+volterra/data/CSD/Test/Snow",
    "csd_test_gt"        : r"E:/restormer+volterra/data/CSD/Test/Gt",
    "classic5_root"      : r"E:/restormer+volterra/data/classic5",
    "live1_root"         : r"E:/restormer+volterra/data/LIVE1",
    # optional
    "bsds_val_img"       : r"E:/restormer+volterra/data/BSDS500/images/val",
    "bsds_val_gt"        : r"E:/restormer+volterra/data/BSDS500/ground_truth/val",
    "bsds_test_img"      : r"E:/restormer+volterra/data/BSDS500/images/test",
    "bsds_test_gt"       : r"E:/restormer+volterra/data/BSDS500/ground_truth/test",
}

def build_train_dataset(transform):
    print("\n📦 Preparing balanced training datasets (160 each)...")
    ds_list = []
    ds_list.append(subset_dataset(PairedFolderDataset(PATHS["rain100h_train_rain"], PATHS["rain100h_train_gt"], transform), MAX_SAMPLES, seed=SEED+1))
    print(f"Rain100H train : {len(ds_list[-1])}")
    ds_list.append(subset_dataset(PairedFolderDataset(PATHS["rain100l_train_rain"], PATHS["rain100l_train_gt"], transform), MAX_SAMPLES, seed=SEED+2))
    print(f"Rain100L train : {len(ds_list[-1])}")
    ds_list.append(subset_dataset(PairedCSVDataset(PATHS["gopro_train_csv"], transform), MAX_SAMPLES, seed=SEED+3))
    print(f"GoPro train    : {len(ds_list[-1])}")
    ds_list.append(subset_dataset(PairedCSVDataset(PATHS["sidd_train_csv"], transform), MAX_SAMPLES, seed=SEED+4))
    print(f"SIDD train     : {len(ds_list[-1])}")
    ds_list.append(subset_dataset(PairedFolderDataset(PATHS["csd_train_snow"], PATHS["csd_train_gt"], transform), MAX_SAMPLES, seed=SEED+5))
    print(f"CSD train      : {len(ds_list[-1])}")
    ds_list.append(subset_dataset(PairedFolderDataset(PATHS["bsds_train_img"], PATHS["bsds_train_gt"], transform), MAX_SAMPLES, seed=SEED+6))
    print(f"BSDS500 train  : {len(ds_list[-1])}")
    total = sum(len(d) for d in ds_list)
    print(f"📊 Total balanced training images = {total}\n")
    return ConcatDataset(ds_list)

def build_test_datasets(transform):
    suites = {}
    def add_if_exists(name, ds):
        try:
            if len(ds) > 0:
                suites[name] = ds
                print(f"  • {name:16s}: {len(ds)}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    print("🧪 Preparing test datasets (unseen wrt training samples)...")
    if os.path.isdir(PATHS["rain100h_test_rain"]) and os.path.isdir(PATHS["rain100h_test_gt"]):
        add_if_exists("Rain100H(test)", PairedFolderDataset(PATHS["rain100h_test_rain"], PATHS["rain100h_test_gt"], transform))
    if os.path.isdir(PATHS["rain100l_test_rain"]) and os.path.isdir(PATHS["rain100l_test_gt"]):
        add_if_exists("Rain100L(test)", PairedFolderDataset(PATHS["rain100l_test_rain"], PATHS["rain100l_test_gt"], transform))
    if os.path.isfile(PATHS["gopro_test_csv"]):
        add_if_exists("GoPro(test)", PairedCSVDataset(PATHS["gopro_test_csv"], transform))
    if os.path.isdir(PATHS["hide_test_blur"]) and os.path.isdir(PATHS["hide_test_gt"]):
        add_if_exists("HIDE", PairedFolderDataset(PATHS["hide_test_blur"], PATHS["hide_test_gt"], transform))
    if os.path.isfile(PATHS["sidd_test_csv"]):
        add_if_exists("SIDD(test)", PairedCSVDataset(PATHS["sidd_test_csv"], transform))
    if os.path.isdir(PATHS["csd_test_snow"]) and os.path.isdir(PATHS["csd_test_gt"]):
        add_if_exists("CSD(Test)", PairedFolderDataset(PATHS["csd_test_snow"], PATHS["csd_test_gt"], transform))
    if os.path.isdir(PATHS["classic5_root"]):
        add_if_exists("Classic5", Classic5Dataset(PATHS["classic5_root"], transform))
    if os.path.isdir(PATHS["live1_root"]):
        add_if_exists("LIVE1", LIVE1Dataset(PATHS["live1_root"], transform))
    if os.path.isdir(PATHS["bsds_val_img"]) and os.path.isdir(PATHS["bsds_val_gt"]):
        add_if_exists("BSDS500(val)", PairedFolderDataset(PATHS["bsds_val_img"], PATHS["bsds_val_gt"], transform))
    if os.path.isdir(PATHS["bsds_test_img"]) and os.path.isdir(PATHS["bsds_test_gt"]):
        add_if_exists("BSDS500(test)", PairedFolderDataset(PATHS["bsds_test_img"], PATHS["bsds_test_gt"], transform))
    print()
    return suites

# ─────────────────────────────────────────────────────────────────────
# 학습/평가
# ─────────────────────────────────────────────────────────────────────
def tensor_to_numpy(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

@torch.no_grad()
def evaluate_dataset(model, loader):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        out = model(inp).clamp(0, 1)
        outs = out.detach().cpu().numpy()
        gts  = tgt.detach().cpu().numpy()
        for o, g in zip(outs, gts):
            o = np.transpose(o, (1, 2, 0))
            g = np.transpose(g, (1, 2, 0))
            total_psnr += compute_psnr(g, o, data_range=1.0)
            total_ssim += compute_ssim(g, o, data_range=1.0, channel_axis=-1)
            n += 1
    return (total_psnr / max(1, n), total_ssim / max(1, n), n)

def eval_all(model, epoch_tag):
    transform = get_transform(EPOCHS)  # 테스트는 최종 해상도로
    suites = build_test_datasets(transform)
    print(f"🔎 Evaluating (epoch {epoch_tag}) ...")
    for name, ds in suites.items():
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        psnr, ssim, n = evaluate_dataset(model, loader)
        print(f"  [{name:12s}] N={n:4d} | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}")
    print("")

def train_and_maybe_eval():
    model = MRVNetUNet().to(DEVICE)

    if RESUME_PATH and os.path.exists(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location="cpu"))
        print(f"✅ Resumed from: {RESUME_PATH}")

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.L1Loss()

    psnr_per_epoch, ssim_per_epoch = [], []
    overall_start_time = time.time()

    print("\n🚀 Training starts\n")
    for epoch in range(START_EPOCH, EPOCHS + 1):
        transform = get_transform(epoch)
        train_dataset = build_train_dataset(transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=4, pin_memory=True, drop_last=False)

        model.train()
        total_loss = total_psnr = total_ssim = 0.0
        count = 0
        start_time = time.time()
        loop = tqdm(train_loader, desc=f"[Epoch {epoch}/{EPOCHS}]")

        for i, (inputs, targets) in enumerate(loop):
            iter_start = time.time()

            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with autocast(device_type='cuda', enabled=(DEVICE.type=='cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            if DEVICE.type == 'cuda':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            total_loss += loss.item()

            # Train-batch metrics (미니배치 추정)
            outs = outputs.detach().clamp(0, 1).cpu().numpy()
            gts  = targets.detach().cpu().numpy()
            for out, gt in zip(outs, gts):
                out = np.transpose(out, (1, 2, 0))
                gt  = np.transpose(gt,  (1, 2, 0))
                total_psnr += compute_psnr(gt, out, data_range=1.0)
                total_ssim += compute_ssim(gt, out, data_range=1.0, channel_axis=-1)
                count += 1

            # 진행률/ETA/경과시간
            iter_time = time.time() - iter_start
            remain_iters = len(train_loader) - (i + 1)
            remain_time_str = time.strftime('%H:%M:%S', time.gmtime(remain_iters * iter_time))
            progress = (i + 1) / len(train_loader) * 100
            elapsed_training_time_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - overall_start_time))
            loop.set_postfix({
                "loss": f"{loss.item():.4f}",
                "prog%": f"{progress:4.1f}",
                "ETA": remain_time_str,
                "Elapsed": elapsed_training_time_str
            })

        avg_loss = total_loss / max(1, len(train_loader))
        avg_psnr = total_psnr / max(1, count)
        avg_ssim = total_ssim / max(1, count)
        psnr_per_epoch.append(avg_psnr); ssim_per_epoch.append(avg_ssim)

        epoch_time_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print(f"\n📣 Epoch {epoch:03d} | Loss={avg_loss:.4f} | PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f} | Elapsed={epoch_time_str}")

        # 체크포인트 저장
        ckpt_name = f"epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, ckpt_name))

        # 주기적 테스트
        if EVAL_EVERY and (epoch % EVAL_EVERY == 0 or epoch == EPOCHS):
            eval_all(model, epoch)

    # 전체 요약 & 최종 테스트
    print("\n[INFO] All Epoch PSNR / SSIM (train-estimate):")
    for i in range(len(psnr_per_epoch)):
        print(f"  Epoch {i + START_EPOCH}: PSNR={psnr_per_epoch[i]:.2f} dB | SSIM={ssim_per_epoch[i]:.4f}")

    print("\n🧪 Final Evaluation on Unseen Test Sets")
    eval_all(model, EPOCHS)

# ─────────────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_maybe_eval()
