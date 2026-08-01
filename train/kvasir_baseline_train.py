"""
Polyp Segmentation Baseline — PVTv2-B2 + Lightweight Decoder
=============================================================
Trains on Kvasir-SEG only. Same architecture as Kvasir-SEG seg model.
This is Client 2's standalone baseline for comparison against FL model.

LOCAL STRUCTURE:
  ./
  ├── Kvasir-SEG/
  └── checkpoints/kvasir_checkpoints/        ← created automatically

USAGE:
  python train/kvasir_baseline_train.py
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
KVASIR_ROOT    = str(PROJECT_ROOT / "Kvasir-SEG")
IMAGE_DIR      = os.path.join(KVASIR_ROOT, "images")
MASK_DIR       = os.path.join(KVASIR_ROOT, "masks")

DRIVE_CKPT_DIR = str(PROJECT_ROOT / "checkpoints" / "kvasir_checkpoints")
LOCAL_CKPT_DIR = DRIVE_CKPT_DIR

TOTAL_EPOCHS   = 50
BATCH_SIZE     = 6
LEARNING_RATE  = 1e-4
BACKBONE_LR    = 1e-5
IMAGE_SIZE     = 352
VAL_SPLIT      = 0.1
KV_TEST_SPLIT  = 0.1
KV_SPLIT_SEED  = 42
KV_SPLIT_FILE  = str(PROJECT_ROOT / "checkpoints" / "fl_checkpoints_original" / "kvasir_split_original.json")
ALT_SPLIT_FILE = str(PROJECT_ROOT / "checkpoints" / "kvasir_checkpoints" / "kvasir_split_original.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


# ─────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────
STATE_FILE = os.path.join(DRIVE_CKPT_DIR, "polyp_state.json")

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        print(f"Resuming from epoch {state['last_completed_epoch']}")
        return state
    print("No checkpoint found — starting fresh")
    return {
        "last_completed_epoch": 0,
        "best_val_loss":        float("inf"),
        "best_dice":            0.0,
        "train_loss_history":   [],
        "val_loss_history":     [],
        "dice_history":         [],
    }

def save_state(state: dict):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


def find_mask_path(mask_dir: str, stem: str) -> str:
    for ext in [".jpg", ".png"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return os.path.join(mask_dir, stem + ".jpg")


class KvasirDataset(Dataset):
    def __init__(self, records: list, augment: bool = False):
        self.records = records
        self.augment = augment

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

            if self.augment:
                if np.random.rand() > 0.5:
                    img_arr  = np.fliplr(img_arr).copy()
                    mask_arr = np.fliplr(mask_arr).copy()
                if np.random.rand() > 0.5:
                    img_arr  = np.flipud(img_arr).copy()
                    mask_arr = np.flipud(mask_arr).copy()

            img_tensor  = torch.from_numpy(normalize(img_arr)).float()
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()
            return img_tensor, mask_tensor, mask_tensor

        except Exception as e:
            print(f"  Warning: failed to load {r['img']}: {e}")
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dummy_mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, dummy_mask, dummy_mask


def build_kvasir_split() -> dict:
    split_path = KV_SPLIT_FILE if os.path.exists(KV_SPLIT_FILE) else ALT_SPLIT_FILE
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"Loaded existing Kvasir (original) split — "
              f"train={len(split['train'])} val={len(split['val'])} "
              f"test={len(split['test'])} (held out, untouched)")
        return split

    print("No Kvasir (original) split file found — creating one now (first run only).")

    supported = {".jpg", ".jpeg", ".png"}
    all_stems = sorted({
        Path(fname).stem
        for fname in os.listdir(IMAGE_DIR)
        if Path(fname).suffix.lower() in supported
    })

    rng = np.random.RandomState(KV_SPLIT_SEED)
    rng.shuffle(all_stems)

    n      = len(all_stems)
    n_test = max(1, int(round(n * KV_TEST_SPLIT)))
    n_val  = max(1, int(round(n * VAL_SPLIT)))

    test_stems  = all_stems[:n_test]
    val_stems   = all_stems[n_test:n_test + n_val]
    train_stems = all_stems[n_test + n_val:]

    split = {"train": train_stems, "val": val_stems, "test": test_stems}

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"  Original Kvasir images : {n}")
    print(f"    train stems : {len(train_stems)}")
    print(f"    val stems   : {len(val_stems)}")
    print(f"    test stems  : {len(test_stems)}  (reserved — never loaded here)")
    print(f"  Split saved to: {split_path}")

    return split


def build_kvasir_records(split_key: str, split: dict) -> list:
    allowed = set(split[split_key])
    records = []
    supported = {".jpg", ".jpeg", ".png"}
    for fname in sorted(os.listdir(IMAGE_DIR)):
        if Path(fname).suffix.lower() not in supported:
            continue
        stem = Path(fname).stem
        if stem not in allowed:
            continue
        records.append({
            "img":  os.path.join(IMAGE_DIR, fname),
            "mask": find_mask_path(MASK_DIR, stem),
        })
    return records


def get_dataloaders():
    split   = build_kvasir_split()
    train_r = build_kvasir_records("train", split)
    val_r   = build_kvasir_records("val",   split)

    print(f"Total Kvasir images: {len(train_r) + len(val_r) + len(split['test'])}")
    print(f"Train: {len(train_r)}  Val: {len(val_r)}  (Test stems held out: {len(split['test'])})")

    train_loader = DataLoader(
        KvasirDataset(train_r, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        KvasirDataset(val_r, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2
    )
    return train_loader, val_loader


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
            pretrained=True,
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
        mask_pred = self.seg_head(out)
        edge_pred = self.edge_head(out)
        return mask_pred, edge_pred, p4


# ─────────────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────────────

def bce_iou_loss(pred, target, eps=1e-6):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred_sig = torch.sigmoid(pred)
    inter    = (pred_sig * target).sum(dim=(2, 3))
    union    = (pred_sig + target - pred_sig * target).sum(dim=(2, 3))
    iou_loss = 1 - (inter + eps) / (union + eps)
    return bce + iou_loss.mean()

def combined_loss(mask_pred, edge_pred, mask_gt, edge_gt):
    seg_loss  = bce_iou_loss(mask_pred, mask_gt)
    edge_loss = bce_iou_loss(edge_pred, edge_gt)
    return seg_loss + 0.5 * edge_loss, seg_loss.item(), edge_loss.item()


# ─────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────

def compute_dice(pred, gt, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    gt       = gt.float()
    inter    = (pred_bin * gt).sum(dim=(2, 3))
    union    = pred_bin.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
    dice     = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


def compute_iou(pred, gt, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    gt       = gt.float()
    inter    = (pred_bin * gt).sum(dim=(2, 3))
    union    = pred_bin.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    iou      = (inter + eps) / (union + eps)
    return iou.mean().item()


# ─────────────────────────────────────────────────────
# MODEL LOAD / SAVE
# ─────────────────────────────────────────────────────

def load_model(state: dict) -> nn.Module:
    model = CamouflageSegNet().to(DEVICE)
    ckpt_path = os.path.join(DRIVE_CKPT_DIR, "latest.pth")

    if state["last_completed_epoch"] > 0 and os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print("Weights loaded.")
    else:
        print("Starting with pretrained PVTv2-B2 encoder.")

    return model


def save_model(model, state, epoch, val_loss, dice, iou, is_best=False):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "dice": dice,
        "iou": iou,
    }
    torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "latest.pth"))
    if epoch % 5 == 0:
        torch.save(payload, os.path.join(DRIVE_CKPT_DIR, f"epoch_{epoch}.pth"))
    if is_best:
        torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "best.pth"))
        print(f"✓ Best model saved (Dice={dice:.4f} IoU={iou:.4f})")


# ─────────────────────────────────────────────────────
# TRAIN / VALIDATE
# ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    steps = 0

    for step, (img, mask, edge) in enumerate(loader):
        img  = img.to(DEVICE)
        mask = mask.to(DEVICE)
        edge = edge.to(DEVICE)

        mask_pred, edge_pred, _ = model(img)
        loss, seg_l, edge_l = combined_loss(mask_pred, edge_pred, mask, edge)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if (step + 1) % 20 == 0:
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} "
                  f"(seg={seg_l:.4f} edge={edge_l:.4f})")

    return total_loss / max(steps, 1)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou  = 0.0
    steps = 0

    for img, mask, edge in loader:
        img  = img.to(DEVICE)
        mask = mask.to(DEVICE)
        edge = edge.to(DEVICE)

        mask_pred, edge_pred, _ = model(img)
        loss, _, _ = combined_loss(mask_pred, edge_pred, mask, edge)
        total_loss += loss.item()
        total_dice += compute_dice(mask_pred, mask)
        total_iou  += compute_iou(mask_pred, mask)
        steps += 1

    return (
        total_loss / max(steps, 1),
        total_dice / max(steps, 1),
        total_iou  / max(steps, 1),
    )


def main():
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

    if not os.path.exists(IMAGE_DIR):
        print(f"\nERROR: {IMAGE_DIR} not found.")
        return

    state = load_state()
    last_epoch = state["last_completed_epoch"]

    if last_epoch >= TOTAL_EPOCHS:
        print(f"\nTraining complete — {TOTAL_EPOCHS} epochs done.")
        print(f"Best model    : {DRIVE_CKPT_DIR}/best.pth")
        print(f"Best Dice     : {state['best_dice']:.4f}")
        return

    start_epoch = last_epoch + 1
    end_epoch   = TOTAL_EPOCHS

    print(f"\n{'='*55}")
    print(f"  Polyp Baseline — PVTv2-B2 on Kvasir-SEG")
    print(f"  This session : epochs {start_epoch} → {end_epoch}")
    print(f"  Total target : {TOTAL_EPOCHS} epochs")
    print(f"{'='*55}\n")

    model = load_model(state)
    train_loader, val_loader = get_dataloaders()

    backbone_params = list(model.encoder.parameters())
    decoder_params  = [p for n, p in model.named_parameters() if "encoder" not in n]

    optimizer = AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": decoder_params,  "lr": LEARNING_RATE},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    for _ in range(last_epoch):
        scheduler.step()

    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n── Epoch {epoch}/{TOTAL_EPOCHS} ──")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, epoch)
        val_loss, dice, iou = validate(model, val_loader)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = dice > state["best_dice"]

        print(f"\n  Train Loss : {train_loss:.4f}")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Dice       : {dice:.4f} {'← best' if is_best else ''}")
        print(f"  IoU        : {iou:.4f}")
        print(f"  Time       : {elapsed/60:.1f} min")

        state["last_completed_epoch"] = epoch
        state["train_loss_history"].append(round(train_loss, 4))
        state["val_loss_history"].append(round(val_loss, 4))
        state["dice_history"].append(round(dice, 4))

        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:
            save_model(model, state, epoch, val_loss, dice, iou, is_best)
            save_state(state)
            print(f"✓ Checkpoint saved (epoch {epoch})")

    print(f"\n{'='*55}")
    print(f"  Session complete: epochs {start_epoch}–{end_epoch} done")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()