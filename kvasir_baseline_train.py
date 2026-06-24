"""
Polyp Segmentation Baseline — PVTv2-B2 + Lightweight Decoder
=============================================================
Trains on Kvasir-SEG only. Same architecture as Kvasir-SEG seg model.
This is Client 2's standalone baseline for comparison against FL model.

Checkpoint-based: each run trains EPOCHS_PER_RUN epochs and saves to Drive.
Run again next session to continue.

INSTALL:
  pip install torch torchvision timm pillow numpy

DRIVE STRUCTURE:
  MyDrive/COD_Project/
  ├── kvasir-seg.zip
  └── polyp_checkpoints/        ← created automatically

SETUP (run once at top of Colab notebook):
  import zipfile, os
  zip_path = "kvasir-seg.zip"
  extract_to = "kvasir-seg"
  if not os.path.exists(extract_to):
      with zipfile.ZipFile(zip_path, 'r') as z:
          z.extractall(extract_to)
      print("Extracted.")

USAGE (run each Colab session):
  !python train_polyp_baseline.py
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
# Kvasir-SEG structure after extraction:
#   kvasir-seg/images/   ← .jpg files
#   kvasir-seg/masks/    ← .jpg files (same names)
KVASIR_ROOT    = "Kvasir-SEG"
IMAGE_DIR      = os.path.join(KVASIR_ROOT, "images")
MASK_DIR       = os.path.join(KVASIR_ROOT, "masks")

DRIVE_CKPT_DIR = "kvasir_checkpoints"
LOCAL_CKPT_DIR = "kvasir_checkpoints"

TOTAL_EPOCHS   = 50
BATCH_SIZE     = 6
LEARNING_RATE  = 1e-4
BACKBONE_LR    = 1e-5
IMAGE_SIZE     = 352
VAL_SPLIT      = 0.1
# EPOCHS_PER_RUN = 5

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
    return img.transpose(2, 0, 1)   # HWC → CHW


def find_mask_path(mask_dir: str, stem: str) -> str:
    """
    Kvasir masks use same filename as images.
    Try .jpg first, then .png as fallback.
    """
    for ext in [".jpg", ".png"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return os.path.join(mask_dir, stem + ".jpg")  # will fail gracefully


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

            # Augmentation
            if self.augment:
                if np.random.rand() > 0.5:
                    img_arr  = np.fliplr(img_arr).copy()
                    mask_arr = np.fliplr(mask_arr).copy()
                if np.random.rand() > 0.5:
                    img_arr  = np.flipud(img_arr).copy()
                    mask_arr = np.flipud(mask_arr).copy()

            img_tensor  = torch.from_numpy(normalize(img_arr)).float()
            mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0).float()

            # edge_tensor = mask_tensor (fallback — no edge maps in Kvasir)
            return img_tensor, mask_tensor, mask_tensor

        except Exception as e:
            print(f"  Warning: failed to load {r['img']}: {e}")
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dummy_mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, dummy_mask, dummy_mask


def build_records(image_dir: str, mask_dir: str) -> list:
    records = []
    supported = {".jpg", ".jpeg", ".png"}
    for fname in sorted(os.listdir(image_dir)):
        ext = Path(fname).suffix.lower()
        if ext not in supported:
            continue
        stem = Path(fname).stem
        records.append({
            "img":  os.path.join(image_dir, fname),
            "mask": find_mask_path(mask_dir, stem),
        })
    return records


def get_dataloaders():
    records = build_records(IMAGE_DIR, MASK_DIR)
    print(f"Total Kvasir images: {len(records)}")

    val_n   = max(1, int(len(records) * VAL_SPLIT))
    train_r = records[:-val_n]
    val_r   = records[-val_n:]
    print(f"Train: {len(train_r)}  Val: {len(val_r)}")

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
# MODEL — identical to COD seg model
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
    """
    Identical architecture to COD seg model.
    Same weights can be used as starting point for FL.
    """
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
# METRICS — Dice (standard polyp metric)
# ─────────────────────────────────────────────────────

def compute_dice(pred, gt, eps=1e-6):
    """
    Dice coefficient — standard polyp segmentation metric.
    Higher is better. Range [0, 1].
    """
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


# def save_model(model, state, epoch, val_loss, dice, iou, is_best=False):
#     os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
#     payload = {
#         "model_state": model.state_dict(),
#         "epoch":       epoch,
#         "val_loss":    val_loss,
#         "dice":        dice,
#         "iou":         iou,
#     }
#     torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "latest.pth"))
#     torch.save(payload, os.path.join(DRIVE_CKPT_DIR, f"epoch_{epoch}.pth"))
#     if is_best:
#         torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "best.pth"))
#         print(f"  ✓ Best model saved (Dice={dice:.4f}  IoU={iou:.4f})")

def save_model(model, state, epoch, val_loss, dice, iou, is_best=False):

    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "dice": dice,
        "iou": iou,
    }

    # Always overwrite latest model
    torch.save(
        payload,
        os.path.join(DRIVE_CKPT_DIR, "latest.pth")
    )

    # Create a permanent checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save(
            payload,
            os.path.join(
                DRIVE_CKPT_DIR,
                f"epoch_{epoch}.pth"
            )
        )

    # Save best model
    if is_best:
        torch.save(
            payload,
            os.path.join(
                DRIVE_CKPT_DIR,
                "best.pth"
            )
        )

        print(
            f"✓ Best model saved (Dice={dice:.4f} IoU={iou:.4f})"
        )


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


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

    # Verify dataset exists
    if not os.path.exists(IMAGE_DIR):
        print(f"\nERROR: {IMAGE_DIR} not found.")
        print("Run this first in your Colab notebook:")
        print("  import zipfile")
        print("  with zipfile.ZipFile('kvasir-seg.zip') as z:")
        print("      z.extractall('kvasir-seg')")
        return

    state = load_state()
    last_epoch = state["last_completed_epoch"]

    if last_epoch >= TOTAL_EPOCHS:
        print(f"\nTraining complete — {TOTAL_EPOCHS} epochs done.")
        print(f"Best model    : {DRIVE_CKPT_DIR}/best.pth")
        print(f"Best Dice     : {state['best_dice']:.4f}")
        return

    # start_epoch = last_epoch + 1
    # end_epoch   = min(last_epoch + EPOCHS_PER_RUN, TOTAL_EPOCHS)

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

        # if is_best:
        #     state["best_dice"]     = dice
        #     state["best_val_loss"] = val_loss

        # save_model(model, state, epoch, val_loss, dice, iou, is_best)

        # state["last_completed_epoch"] = epoch
        # state["train_loss_history"].append(round(train_loss, 4))
        # state["val_loss_history"].append(round(val_loss, 4))
        # state["dice_history"].append(round(dice, 4))
        # save_state(state)

        # print(f"  ✓ Checkpoint saved (epoch {epoch})")

        state["last_completed_epoch"] = epoch

        state["train_loss_history"].append(round(train_loss, 4))
        state["val_loss_history"].append(round(val_loss, 4))
        state["dice_history"].append(round(dice, 4))

        if epoch % 5 == 0 or epoch == TOTAL_EPOCHS:

            save_model(
                model,
                state,
                epoch,
                val_loss,
                dice,
                iou,
                is_best
            )

            save_state(state)

            print(f"✓ Checkpoint saved (epoch {epoch})")

    print(f"\n{'='*55}")
    print(f"  Session complete: epochs {start_epoch}–{end_epoch} done")
    if end_epoch < TOTAL_EPOCHS:
        print(f"  Run again next session to continue from epoch {end_epoch + 1}")
    else:
        print(f"  ALL {TOTAL_EPOCHS} EPOCHS COMPLETE")
        print(f"  Best model : {DRIVE_CKPT_DIR}/best.pth")
        print(f"  Best Dice  : {state['best_dice']:.4f}")

    print(f"\n  Dice history:")
    for i, d in enumerate(state["dice_history"], 1):
        bar = "█" * int(d * 30)
        print(f"    Epoch {i:>2}: {d:.4f} {bar}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()