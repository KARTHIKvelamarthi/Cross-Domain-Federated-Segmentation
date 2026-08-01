"""
Cross-Domain Baseline Evaluation
=================================
Evaluates single-domain baseline models (COD10K-only and Kvasir-only)
live from disk across both in-domain and cross-domain test sets.

Matrix:
                    COD10K Test     Kvasir Test
  COD10K-only       ✓ (in-domain)   ✓ (cross-domain)
  Kvasir-only       ✓ (cross-domain)✓ (in-domain)

CHECKPOINTS NEEDED:
  checkpoints/cod10k_checkpoints/best.pth
  checkpoints/kvasir_checkpoints/best.pth

DATASETS NEEDED:
  COD10K-v3/Test/Image/
  COD10K-v3/Test/GT_Object/
  Kvasir-SEG/images/
  Kvasir-SEG/masks/
  checkpoints/fl_checkpoints_original/kvasir_split_original.json

USAGE:
  python compare/compare_baselines.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import timm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
PROJECT_ROOT       = Path(__file__).resolve().parent.parent

COD_TEST_IMAGE_DIR = str(PROJECT_ROOT / "COD10K-v3" / "Test" / "Image")
COD_TEST_MASK_DIR  = str(PROJECT_ROOT / "COD10K-v3" / "Test" / "GT_Object")

KVASIR_IMAGE_DIR   = str(PROJECT_ROOT / "Kvasir-SEG" / "images")
KVASIR_MASK_DIR    = str(PROJECT_ROOT / "Kvasir-SEG" / "masks")
KVASIR_VAL_SPLIT   = 0.1

BASELINE_COD_CKPT  = str(PROJECT_ROOT / "checkpoints" / "cod10k_checkpoints" / "best.pth")
BASELINE_KV_CKPT   = str(PROJECT_ROOT / "checkpoints" / "kvasir_checkpoints" / "best.pth")

KV_SPLIT_FILE      = str(PROJECT_ROOT / "checkpoints" / "fl_checkpoints_original" / "kvasir_split_original.json")
RESULTS_PATH       = str(PROJECT_ROOT / "results" / "baseline_results.json")

IMAGE_SIZE  = 352
BATCH_SIZE  = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


class SegDataset(Dataset):
    def __init__(self, records: list):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img  = Image.open(r["img"]).convert("RGB").resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            mask = Image.open(r["mask"]).convert("L").resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

            img_arr  = np.array(img)
            mask_arr = (np.array(mask) > 128).astype(np.float32)

            return (
                torch.from_numpy(normalize(img_arr)).float(),
                torch.from_numpy(mask_arr).unsqueeze(0).float(),
                r.get("name", str(idx)),
            )
        except Exception as e:
            print(f"  Warning: failed to load {r['img']}: {e}")
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dm    = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, dm, "error"


def find_mask_path(mask_dir: str, stem: str) -> str:
    for ext in [".jpg", ".png"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return os.path.join(mask_dir, stem + ".jpg")


def build_cod_test_records() -> list:
    records = []
    for fname in sorted(os.listdir(COD_TEST_IMAGE_DIR)):
        if not fname.endswith(".jpg"):
            continue
        stem = fname.replace(".jpg", "")
        mask_path = os.path.join(COD_TEST_MASK_DIR, stem + ".png")
        if os.path.exists(mask_path):
            records.append({
                "img":  os.path.join(COD_TEST_IMAGE_DIR, fname),
                "mask": mask_path,
                "name": stem,
            })
    return records


def build_kvasir_test_records() -> list:
    split_file = KV_SPLIT_FILE if os.path.exists(KV_SPLIT_FILE) else str(PROJECT_ROOT / "checkpoints" / "kvasir_checkpoints" / "kvasir_split_original.json")
    if os.path.exists(split_file):
        with open(split_file) as f:
            split = json.load(f)
        test_stems = set(split["test"])
        records = []
        supported = {".jpg", ".jpeg", ".png"}
        for fname in sorted(os.listdir(KVASIR_IMAGE_DIR)):
            if Path(fname).suffix.lower() not in supported:
                continue
            stem = Path(fname).stem
            if stem in test_stems:
                records.append({
                    "img":  os.path.join(KVASIR_IMAGE_DIR, fname),
                    "mask": find_mask_path(KVASIR_MASK_DIR, stem),
                    "name": stem,
                })
        return records
    else:
        all_records = []
        supported = {".jpg", ".jpeg", ".png"}
        for fname in sorted(os.listdir(KVASIR_IMAGE_DIR)):
            if Path(fname).suffix.lower() not in supported:
                continue
            stem = Path(fname).stem
            all_records.append({
                "img":  os.path.join(KVASIR_IMAGE_DIR, fname),
                "mask": find_mask_path(KVASIR_MASK_DIR, stem),
                "name": stem,
            })
        val_n = max(1, int(len(all_records) * KVASIR_VAL_SPLIT))
        return all_records[-val_n:]


def get_loader(records: list) -> DataLoader:
    return DataLoader(
        SegDataset(records),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )


# ─────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class CamouflageSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "pvt_v2_b2",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        enc_channels = [64, 128, 320, 512]

        self.lat4 = ConvBnRelu(enc_channels[3], 256)
        self.lat3 = ConvBnRelu(enc_channels[2], 256)
        self.lat2 = ConvBnRelu(enc_channels[1], 128)
        self.lat1 = ConvBnRelu(enc_channels[0], 64)

        self.merge3 = ConvBnRelu(256 + 256, 256)
        self.merge2 = ConvBnRelu(256 + 128, 128)
        self.merge1 = ConvBnRelu(128 + 64,  64)

        self.seg_head = nn.Sequential(
            ConvBnRelu(64, 64),
            nn.Conv2d(64, 1, 1),
        )
        self.edge_head = nn.Sequential(
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        f1, f2, f3, f4 = self.encoder(x)

        p4 = self.lat4(f4)
        p3 = self.lat3(f3)
        p2 = self.lat2(f2)
        p1 = self.lat1(f1)

        p3 = self.merge3(torch.cat([
            F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False), p3
        ], dim=1))
        p2 = self.merge2(torch.cat([
            F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False), p2
        ], dim=1))
        p1 = self.merge1(torch.cat([
            F.interpolate(p2, size=p1.shape[2:], mode="bilinear", align_corners=False), p1
        ], dim=1))

        out = F.interpolate(p1, size=(H, W), mode="bilinear", align_corners=False)
        return self.seg_head(out), self.edge_head(out), p4


def load_model(ckpt_path: str, label: str) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{label} not found: {ckpt_path}")
    model = CamouflageSegNet().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded {label} ({os.path.basename(ckpt_path)})")
    return model


# ─────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────

def compute_dice(pred_bin: torch.Tensor, gt: torch.Tensor, eps=1e-6) -> float:
    inter = (pred_bin * gt).sum(dim=(2, 3))
    union = pred_bin.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def compute_iou(pred_prob: torch.Tensor, gt: torch.Tensor, eps=1e-6) -> float:
    inter = (pred_prob * gt).sum(dim=(2, 3))
    union = (pred_prob + gt - pred_prob * gt).sum(dim=(2, 3))
    return ((inter + eps) / (union + eps)).mean().item()

def compute_mae(pred_prob: torch.Tensor, gt: torch.Tensor) -> float:
    return (pred_prob - gt).abs().mean().item()

def compute_fmeasure(pred_bin: torch.Tensor, gt: torch.Tensor,
                     beta2: float = 0.3, eps=1e-6) -> float:
    tp   = (pred_bin * gt).sum(dim=(2, 3))
    fp   = (pred_bin * (1 - gt)).sum(dim=(2, 3))
    fn   = ((1 - pred_bin) * gt).sum(dim=(2, 3))
    prec = (tp + eps) / (tp + fp + eps)
    rec  = (tp + eps) / (tp + fn + eps)
    fm   = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
    return fm.mean().item()

def compute_smeasure(pred_prob: np.ndarray, gt: np.ndarray,
                     alpha: float = 0.5, eps=1e-6) -> float:
    gt_bool = gt.astype(bool)
    if gt_bool.sum() == 0:
        return 1.0 if pred_prob.max() < 0.5 else 0.0
    x     = pred_prob[gt_bool].mean()
    sigma = pred_prob[gt_bool].std() + eps
    Q_o   = 2.0 * x / (x**2 + 1.0 + sigma + eps)
    pred_bg  = 1.0 - pred_prob
    x_bg     = pred_bg[~gt_bool].mean()
    sigma_bg = pred_bg[~gt_bool].std() + eps
    Q_b      = 2.0 * x_bg / (x_bg**2 + 1.0 + sigma_bg + eps)
    w_o = gt_bool.sum() / gt_bool.size
    w_b = 1.0 - w_o
    return alpha * Q_o + (1 - alpha) * (w_o * Q_o + w_b * Q_b)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             model_label: str, domain_label: str) -> dict:
    model.eval()

    all_dice, all_iou, all_mae, all_fm, all_sm = [], [], [], [], []

    for img, mask, _ in loader:
        img  = img.to(DEVICE)
        mask = mask.to(DEVICE)

        mask_pred, _, _ = model(img)
        pred_prob = torch.sigmoid(mask_pred)
        pred_bin  = (pred_prob > 0.5).float()

        all_dice.append(compute_dice(pred_bin, mask))
        all_iou.append(compute_iou(pred_prob, mask))
        all_mae.append(compute_mae(pred_prob, mask))
        all_fm.append(compute_fmeasure(pred_bin, mask))

        prob_np = pred_prob.squeeze(1).cpu().numpy()
        mask_np = mask.squeeze(1).cpu().numpy()
        for i in range(prob_np.shape[0]):
            all_sm.append(compute_smeasure(prob_np[i], mask_np[i]))

    return {
        "model":     model_label,
        "domain":    domain_label,
        "Dice":      round(float(np.mean(all_dice)), 4),
        "IoU":       round(float(np.mean(all_iou)),  4),
        "F-measure": round(float(np.mean(all_fm)),   4),
        "S-measure": round(float(np.mean(all_sm)),   4),
        "MAE":       round(float(np.mean(all_mae)),  4),
    }


def print_results_table(all_results: dict):
    models = ["COD10K-only", "Kvasir-only"]

    print(f"\n{'='*80}")
    print(f"  BASELINE CROSS-DOMAIN EVALUATION — RESULTS TABLE")
    print(f"{'='*80}")
    print(f"\n  {'Model':<16} {'COD10K Test':^30} {'Kvasir Test':^30}")
    print(f"  {'':16} {'Dice':>6} {'IoU':>6} {'Fm':>6} {'Sm':>6} {'MAE':>6}  "
          f"{'Dice':>6} {'IoU':>6} {'Fm':>6} {'Sm':>6} {'MAE':>6}")
    print(f"  {'-'*76}")

    for model_name in models:
        cod_r = all_results.get(f"{model_name}_COD10K", {})
        kv_r  = all_results.get(f"{model_name}_Kvasir", {})

        def fmt(r, key):
            return f"{r[key]:6.4f}" if key in r else "  N/A "

        print(f"  {model_name:<16} "
              f"{fmt(cod_r,'Dice')} {fmt(cod_r,'IoU')} "
              f"{fmt(cod_r,'F-measure')} {fmt(cod_r,'S-measure')} {fmt(cod_r,'MAE')}  "
              f"{fmt(kv_r,'Dice')} {fmt(kv_r,'IoU')} "
              f"{fmt(kv_r,'F-measure')} {fmt(kv_r,'S-measure')} {fmt(kv_r,'MAE')}")

    print(f"  {'-'*76}")
    print(f"{'='*80}\n")


def main():
    required = [
        (COD_TEST_IMAGE_DIR, "COD10K Test Images"),
        (COD_TEST_MASK_DIR,  "COD10K Test Masks"),
        (KVASIR_IMAGE_DIR,   "Kvasir Images"),
        (KVASIR_MASK_DIR,    "Kvasir Masks"),
        (BASELINE_COD_CKPT,  "COD10K Baseline Checkpoint"),
        (BASELINE_KV_CKPT,   "Kvasir Baseline Checkpoint"),
    ]
    for path, label in required:
        if not os.path.exists(path):
            print(f"\nERROR: {label} not found:\n  {path}")
            return

    print(f"\n{'='*80}")
    print(f"  Baseline Cross-Domain Evaluation — Live Checkpoints")
    print(f"{'='*80}\n")

    cod_records = build_cod_test_records()
    kv_records  = build_kvasir_test_records()
    print(f"COD10K test images : {len(cod_records)}")
    print(f"Kvasir test images : {len(kv_records)}")

    cod_loader = get_loader(cod_records)
    kv_loader  = get_loader(kv_records)

    print(f"\nLoading baseline models...")
    cod_model = load_model(BASELINE_COD_CKPT, "COD10K-only baseline")
    kv_model  = load_model(BASELINE_KV_CKPT,  "Kvasir-only baseline")

    all_results = {}

    combos = [
        (cod_model, cod_loader, "COD10K-only", "COD10K"),   # in-domain
        (cod_model, kv_loader,  "COD10K-only", "Kvasir"),   # cross-domain
        (kv_model,  cod_loader, "Kvasir-only", "COD10K"),   # cross-domain
        (kv_model,  kv_loader,  "Kvasir-only", "Kvasir"),   # in-domain
    ]

    for model, loader, model_label, domain_label in combos:
        print(f"\n  Evaluating {model_label} on {domain_label} test set...")
        result = evaluate(model, loader, model_label, domain_label)
        key = f"{model_label}_{domain_label}"
        all_results[key] = result
        print(f"    Dice={result['Dice']:.4f}  IoU={result['IoU']:.4f}  "
              f"Fm={result['F-measure']:.4f}  Sm={result['S-measure']:.4f}  "
              f"MAE={result['MAE']:.4f}")

    print_results_table(all_results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Baseline results saved to: {RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
