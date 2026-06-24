"""
Federated Learning — FedAvg on COD10K + Kvasir-SEG
====================================================
Client 1 : COD10K   (camouflaged object detection)
Client 2 : Kvasir   (polyp segmentation)
Server   : FedAvg aggregation

Checkpoint-based: each run trains ROUNDS_PER_RUN FL rounds and saves to Drive.
Run again next session to continue.

DRIVE STRUCTURE:
  MyDrive/COD_Project/
  ├── COD10K-v3.zip
  ├── kvasir-seg.zip
  └── fl_checkpoints/           ← created automatically
      ├── fl_state.json
      ├── global_latest.pth     ← global model after each round
      ├── global_best.pth       ← best global model
      └── global_round_N.pth    ← per-round snapshots

SETUP (run once at top of Colab notebook):
  import zipfile, os

  if not os.path.exists("COD10K-v3"):
      with zipfile.ZipFile("COD10K-v3.zip") as z:
          z.extractall(".")
      print("COD10K extracted.")

  if not os.path.exists("kvasir-seg"):
      with zipfile.ZipFile("kvasir-seg.zip") as z:
          z.extractall("kvasir-seg")
      print("Kvasir extracted.")

USAGE (run each Colab session):
  !python train_federated.py
"""

import os
import copy
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

# Client 1 — COD10K
COD_IMAGE_DIR = "COD10K-v3/Train/Image"
COD_MASK_DIR  = "COD10K-v3/Train/GT_Object"
COD_EDGE_DIR  = "COD10K-v3/Train/GT_Edge"
COD_TEST_IMG  = "COD10K-v3/Test/Image"
COD_TEST_MASK = "COD10K-v3/Test/GT_Object"

# Client 2 — Kvasir-SEG
KVASIR_ROOT   = "Kvasir-SEG"
KV_IMAGE_DIR  = os.path.join(KVASIR_ROOT, "images")
KV_MASK_DIR   = os.path.join(KVASIR_ROOT, "masks")

DRIVE_CKPT_DIR = "fl_checkpoints"

# FL hyperparameters
TOTAL_ROUNDS    = 50    # total FL communication rounds
ROUNDS_PER_RUN  = 5     # rounds per Colab session
LOCAL_EPOCHS    = 1     # each client trains for this many epochs per round
BATCH_SIZE      = 6
LEARNING_RATE   = 1e-4
BACKBONE_LR     = 1e-5
IMAGE_SIZE      = 352
VAL_SPLIT       = 0.1

# FedAvg weights — equal weighting between clients
# Adjust if dataset sizes are very different
CLIENT_WEIGHTS  = [0.75, 0.25]   # [COD10K weight, Kvasir weight]
MU = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


# ─────────────────────────────────────────────────────
# STATE MANAGEMENT
# ─────────────────────────────────────────────────────
STATE_FILE = os.path.join(DRIVE_CKPT_DIR, "fl_state.json")

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        print(f"Resuming from FL round {state['last_completed_round']}")
        return state
    print("No FL checkpoint found — starting fresh")
    return {
        "last_completed_round":  0,
        "best_avg_dice":         0.0,
        # per-round metrics
        "cod_dice_history":      [],
        "kvasir_dice_history":   [],
        "cod_iou_history":       [],
        "kvasir_iou_history":    [],
        "avg_dice_history":      [],
        "client1_loss_history":  [],
        "client2_loss_history":  [],
    }

def save_state(state: dict):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ─────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


# ─────────────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────────────

class CODDataset(Dataset):
    """Client 1 — COD10K camouflaged object dataset."""
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
            edge = Image.open(r["edge"]).convert("L").resize(
                        (IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST) \
                        if os.path.exists(r["edge"]) else mask

            img_arr  = np.array(img)
            mask_arr = (np.array(mask) > 128).astype(np.float32)
            edge_arr = (np.array(edge) > 128).astype(np.float32)

            if self.augment and np.random.rand() > 0.5:
                img_arr  = np.fliplr(img_arr).copy()
                mask_arr = np.fliplr(mask_arr).copy()
                edge_arr = np.fliplr(edge_arr).copy()

            return (
                torch.from_numpy(normalize(img_arr)).float(),
                torch.from_numpy(mask_arr).unsqueeze(0).float(),
                torch.from_numpy(edge_arr).unsqueeze(0).float(),
            )
        except Exception as e:
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dm    = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, dm, dm


class KvasirDataset(Dataset):
    """Client 2 — Kvasir-SEG polyp dataset."""
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

            mask_t = torch.from_numpy(mask_arr).unsqueeze(0).float()
            return (
                torch.from_numpy(normalize(img_arr)).float(),
                mask_t,
                mask_t,   # edge fallback = mask
            )
        except Exception as e:
            dummy = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            dm    = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
            return dummy, dm, dm


def find_mask_path(mask_dir: str, stem: str) -> str:
    for ext in [".jpg", ".png"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return os.path.join(mask_dir, stem + ".jpg")


def build_cod_records():
    records = []
    for fname in sorted(os.listdir(COD_IMAGE_DIR)):
        if not fname.endswith(".jpg"):
            continue
        stem = fname.replace(".jpg", "")
        records.append({
            "img":  os.path.join(COD_IMAGE_DIR, fname),
            "mask": os.path.join(COD_MASK_DIR,  stem + ".png"),
            "edge": os.path.join(COD_EDGE_DIR,  stem + ".png"),
        })
    return records


def build_kvasir_records():
    records = []
    supported = {".jpg", ".jpeg", ".png"}
    for fname in sorted(os.listdir(KV_IMAGE_DIR)):
        if Path(fname).suffix.lower() not in supported:
            continue
        stem = Path(fname).stem
        records.append({
            "img":  os.path.join(KV_IMAGE_DIR, fname),
            "mask": find_mask_path(KV_MASK_DIR, stem),
        })
    return records


def get_client_loaders():
    """
    Returns train/val loaders for both clients.
    Also returns test loaders for evaluation.
    """
    # Client 1 — COD10K
    cod_records = build_cod_records()
    cod_val_n   = max(1, int(len(cod_records) * VAL_SPLIT))
    cod_train   = cod_records[:-cod_val_n]
    cod_val     = cod_records[-cod_val_n:]

    # Client 2 — Kvasir
    kv_records  = build_kvasir_records()
    kv_val_n    = max(1, int(len(kv_records) * VAL_SPLIT))
    kv_train    = kv_records[:-kv_val_n]
    kv_val      = kv_records[-kv_val_n:]

    print(f"Client 1 (COD10K)  — Train: {len(cod_train)}  Val: {len(cod_val)}")
    print(f"Client 2 (Kvasir)  — Train: {len(kv_train)}  Val: {len(kv_val)}")

    c1_train = DataLoader(CODDataset(cod_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    c1_val   = DataLoader(CODDataset(cod_val, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    c2_train = DataLoader(KvasirDataset(kv_train, augment=True),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
    c2_val   = DataLoader(KvasirDataset(kv_val, augment=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return (c1_train, c1_val), (c2_train, c2_val)


# ─────────────────────────────────────────────────────
# MODEL — identical to both baselines
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
        return self.seg_head(out), self.edge_head(out), p4


# ─────────────────────────────────────────────────────
# LOSS & METRICS
# ─────────────────────────────────────────────────────

def bce_iou_loss(pred, target, eps=1e-6):
    bce      = F.binary_cross_entropy_with_logits(pred, target)
    pred_sig = torch.sigmoid(pred)
    inter    = (pred_sig * target).sum(dim=(2, 3))
    union    = (pred_sig + target - pred_sig * target).sum(dim=(2, 3))
    return bce + (1 - (inter + eps) / (union + eps)).mean()

def combined_loss(mask_pred, edge_pred, mask_gt, edge_gt):
    seg_l  = bce_iou_loss(mask_pred, mask_gt)
    edge_l = bce_iou_loss(edge_pred, edge_gt)
    return seg_l + 0.5 * edge_l, seg_l.item(), edge_l.item()

def compute_dice(pred, gt, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    inter    = (pred_bin * gt).sum(dim=(2, 3))
    union    = pred_bin.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def compute_iou(pred, gt, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    inter    = (pred_bin * gt).sum(dim=(2, 3))
    union    = pred_bin.sum(dim=(2, 3)) + gt.sum(dim=(2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


# ─────────────────────────────────────────────────────
# FEDAVG — core aggregation
# ─────────────────────────────────────────────────────

def fedavg(global_weights: dict, client_weights_list: list, weights: list) -> dict:
    """
    FedAvg: weighted average of client model parameters.

    Args:
        global_weights      : current global model state_dict (unused here, kept for reference)
        client_weights_list : list of state_dicts from each client
        weights             : list of floats summing to 1.0 — client contribution weights

    Returns:
        averaged state_dict
    """
    assert abs(sum(weights) - 1.0) < 1e-5, "Client weights must sum to 1.0"
    assert len(client_weights_list) == len(weights)

    averaged = {}
    for key in client_weights_list[0].keys():
        # Weighted sum across clients
        averaged[key] = sum(
            w * client_w[key].float()
            for w, client_w in zip(weights, client_weights_list)
        )
    return averaged


# ─────────────────────────────────────────────────────
# CLIENT TRAINING
# ─────────────────────────────────────────────────────

def client_train(global_model: nn.Module, train_loader: DataLoader,
                 client_name: str, round_num: int) -> tuple:
    """
    Each client receives a copy of the global model,
    trains locally for LOCAL_EPOCHS, returns updated weights + loss.
    """
    # Deep copy — client trains its own local copy
    local_model = copy.deepcopy(global_model).to(DEVICE)
    local_model.train()

    global_params = {
        name: param.detach().clone()
        for name, param in global_model.named_parameters()
    }

    backbone_params = list(local_model.encoder.parameters())
    decoder_params  = [p for n, p in local_model.named_parameters()
                       if "encoder" not in n]

    optimizer = AdamW([
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": decoder_params,  "lr": LEARNING_RATE},
    ], weight_decay=1e-4)

    total_loss = 0.0
    steps = 0

    for epoch in range(LOCAL_EPOCHS):
        for step, (img, mask, edge) in enumerate(train_loader):
            img  = img.to(DEVICE)
            mask = mask.to(DEVICE)
            edge = edge.to(DEVICE)

            mask_pred, edge_pred, _ = local_model(img)

            loss, seg_l, edge_l = combined_loss(
                mask_pred,
                edge_pred,
                mask,
                edge
            )

            prox_term = torch.tensor(
                0.0,
                device=DEVICE
            )

            for name, param in local_model.named_parameters():

                if "encoder" not in name:

                    prox_term += torch.sum(
                        (param - global_params[name]) ** 2
                    )

            loss = loss + (MU / 2.0) * prox_term

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                local_model.parameters(),
                1.0
            )
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if (step + 1) % 30 == 0:
                print(f"    [{client_name}] Round {round_num} | "
                      f"Step {step+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(steps, 1)
    print(f"    [{client_name}] Avg loss: {avg_loss:.4f}")

    # Return local weights and loss
    return local_model.state_dict(), avg_loss


# ─────────────────────────────────────────────────────
# GLOBAL MODEL EVALUATION
# ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_global(global_model: nn.Module, val_loader: DataLoader,
                    client_name: str) -> tuple:
    """Evaluate global model on a client's validation set."""
    global_model.eval()
    total_dice = 0.0
    total_iou  = 0.0
    steps = 0

    for img, mask, edge in val_loader:
        img  = img.to(DEVICE)
        mask = mask.to(DEVICE)

        mask_pred, _, _ = global_model(img)
        total_dice += compute_dice(mask_pred, mask)
        total_iou  += compute_iou(mask_pred, mask)
        steps += 1

    dice = total_dice / max(steps, 1)
    iou  = total_iou  / max(steps, 1)
    print(f"    [Global → {client_name}] Dice: {dice:.4f}  IoU: {iou:.4f}")
    return dice, iou


# ─────────────────────────────────────────────────────
# MODEL LOAD / SAVE
# ─────────────────────────────────────────────────────

def load_global_model(state: dict) -> nn.Module:
    model = CamouflageSegNet().to(DEVICE)
    ckpt_path = os.path.join(DRIVE_CKPT_DIR, "global_latest.pth")

    if state["last_completed_round"] > 0 and os.path.exists(ckpt_path):
        print(f"Loading global model from round {state['last_completed_round']}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print("Global model loaded.")
    else:
        print("Initialising global model with pretrained PVTv2-B2.")

    return model


def save_global_model(model, state, round_num, cod_dice, kv_dice, is_best=False):
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    avg_dice = (cod_dice + kv_dice) / 2.0
    payload  = {
        "model_state": model.state_dict(),
        "round":       round_num,
        "cod_dice":    cod_dice,
        "kv_dice":     kv_dice,
        "avg_dice":    avg_dice,
    }
    torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "global_latest.pth"))
    torch.save(payload, os.path.join(DRIVE_CKPT_DIR, f"global_round_{round_num}.pth"))
    if is_best:
        torch.save(payload, os.path.join(DRIVE_CKPT_DIR, "global_best.pth"))
        print(f"  ✓ Best global model saved "
              f"(COD Dice={cod_dice:.4f} | Kvasir Dice={kv_dice:.4f} | Avg={avg_dice:.4f})")


# ─────────────────────────────────────────────────────
# MAIN FL LOOP
# ─────────────────────────────────────────────────────

def main():
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

    # Verify datasets
    for path, name in [(COD_IMAGE_DIR, "COD10K"), (KV_IMAGE_DIR, "Kvasir-SEG")]:
        if not os.path.exists(path):
            print(f"\nERROR: {name} not found at {path}")
            print("Run the setup cell at the top of your Colab notebook.")
            return

    state      = load_state()
    last_round = state["last_completed_round"]

    if last_round >= TOTAL_ROUNDS:
        print(f"\nFL Training complete — {TOTAL_ROUNDS} rounds done.")
        print(f"Best global model : {DRIVE_CKPT_DIR}/global_best.pth")
        print(f"Best avg Dice     : {state['best_avg_dice']:.4f}")
        return

    start_round = last_round + 1
    end_round   = min(last_round + ROUNDS_PER_RUN, TOTAL_ROUNDS)

    print(f"\n{'='*60}")
    print(f"  Federated Learning — FedProx")
    print(f"  Clients  : COD10K (Client 1) + Kvasir-SEG (Client 2)")
    print(f"  Weights  : {CLIENT_WEIGHTS}")
    print(f"  This session : rounds {start_round} → {end_round}")
    print(f"  Total target : {TOTAL_ROUNDS} rounds")
    print(f"{'='*60}\n")

    # Load data loaders for both clients
    (c1_train, c1_val), (c2_train, c2_val) = get_client_loaders()

    # Load / initialise global model
    global_model = load_global_model(state)

    # ── FL rounds ──────────────────────────────────────
    for round_num in range(start_round, end_round + 1):
        print(f"\n{'─'*60}")
        print(f"  FL Round {round_num}/{TOTAL_ROUNDS}")
        print(f"{'─'*60}")
        t0 = time.time()

        # ── Step 1: broadcast global model to clients ──
        # (in simulation this is just passing the model reference)

        # ── Step 2: client local training ──────────────
        print(f"\n  [Client 1 — COD10K training]")
        c1_weights, c1_loss = client_train(
            global_model, c1_train, "COD10K", round_num)

        print(f"\n  [Client 2 — Kvasir training]")
        c2_weights, c2_loss = client_train(
            global_model, c2_train, "Kvasir", round_num)

        # ── Step 3: FedAvg aggregation ─────────────────
        print(f"\n  [Server — FedAvg aggregation (FedProx clients)]")
        averaged_weights = fedavg(
            global_model.state_dict(),
            [c1_weights, c2_weights],
            CLIENT_WEIGHTS
        )
        global_model.load_state_dict(averaged_weights)
        print(f"  Aggregation complete.")

        # ── Step 4: evaluate global model ──────────────
        print(f"\n  [Evaluation — global model on both domains]")
        cod_dice, cod_iou   = evaluate_global(global_model, c1_val,  "COD10K")
        kv_dice,  kv_iou    = evaluate_global(global_model, c2_val,  "Kvasir")
        avg_dice = (cod_dice + kv_dice) / 2.0

        elapsed = time.time() - t0
        is_best = avg_dice > state["best_avg_dice"]

        print(f"\n  ── Round {round_num} Summary ──")
        print(f"  Client 1 loss (COD)   : {c1_loss:.4f}")
        print(f"  Client 2 loss (Kvasir): {c2_loss:.4f}")
        print(f"  Global Dice  COD10K   : {cod_dice:.4f}  IoU: {cod_iou:.4f}")
        print(f"  Global Dice  Kvasir   : {kv_dice:.4f}  IoU: {kv_iou:.4f}")
        print(f"  Avg Dice              : {avg_dice:.4f} {'← best' if is_best else ''}")
        print(f"  Round time            : {elapsed/60:.1f} min")

        if is_best:
            state["best_avg_dice"] = avg_dice

        save_global_model(global_model, state, round_num, cod_dice, kv_dice, is_best)

        # Update state
        state["last_completed_round"] = round_num
        state["cod_dice_history"].append(round(cod_dice, 4))
        state["kvasir_dice_history"].append(round(kv_dice, 4))
        state["cod_iou_history"].append(round(cod_iou, 4))
        state["kvasir_iou_history"].append(round(kv_iou, 4))
        state["avg_dice_history"].append(round(avg_dice, 4))
        state["client1_loss_history"].append(round(c1_loss, 4))
        state["client2_loss_history"].append(round(c2_loss, 4))
        save_state(state)

        print(f"  ✓ Checkpoint saved (round {round_num})")

    # ── Session summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Session complete: rounds {start_round}–{end_round} done")

    if end_round < TOTAL_ROUNDS:
        print(f"  Run again next session to continue from round {end_round + 1}")
    else:
        print(f"  ALL {TOTAL_ROUNDS} ROUNDS COMPLETE")
        print(f"  Best global model : {DRIVE_CKPT_DIR}/global_best.pth")
        print(f"  Best avg Dice     : {state['best_avg_dice']:.4f}")

    print(f"\n  Avg Dice per round:")
    for i, d in enumerate(state["avg_dice_history"], 1):
        cod = state["cod_dice_history"][i-1]
        kv  = state["kvasir_dice_history"][i-1]
        print(f"    Round {i:>2}: avg={d:.4f}  cod={cod:.4f}  kvasir={kv:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()