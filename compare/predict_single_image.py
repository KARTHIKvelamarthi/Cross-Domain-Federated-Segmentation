"""
Single-Image Inference — visualize predicted mask overlay
===========================================================================
Takes ONE image path, runs it through the trained CamouflageSegNet
checkpoint, and saves/shows the image with the predicted mask drawn
on top (colored overlay + optional contour outline). No dataset,
no ground-truth mask, no metrics — just a visual sanity check.

USAGE:
  python compare/predict_single_image.py --image COD10K-v3/Test/Image/COD10K-CAM-1-Aquatic-3-Crab-46.jpg
  python compare/predict_single_image.py --image sample.jpg --ckpt checkpoints/fl_checkpoints_aug_new/global_best.pth
  python compare/predict_single_image.py --image sample.jpg --out results/overlay.png --threshold 0.5
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
import cv2
from pathlib import Path

# ─────────────────────────────────────────────────────
# CONFIG DEFAULT VALUES
# ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH   = str(PROJECT_ROOT / "COD10K-v3" / "Test" / "Image" / "COD10K-CAM-1-Aquatic-3-Crab-46.jpg")
CKPT_PATH    = str(PROJECT_ROOT / "checkpoints" / "fl_checkpoints_aug_new" / "global_best.pth")
OUT_PATH     = None   # None -> auto-saves to "results/<image_name>_overlay.png"
THRESHOLD    = 0.5
SIDE_BY_SIDE = True   # saves an original | heatmap | overlay comparison image

IMAGE_SIZE   = 352
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────
# MODEL (identical architecture to training/eval scripts)
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


def load_model(ckpt_path: str) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model = CamouflageSegNet().to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint ({os.path.basename(ckpt_path)}) from {ckpt_path}")
    return model


# ─────────────────────────────────────────────────────
# PRE/POST PROCESSING
# ─────────────────────────────────────────────────────
def load_and_preprocess(image_path: str):
    """Returns (model_input_tensor, original_rgb_uint8_array, original_size (W,H))."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    orig_rgb = np.array(img)

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # CHW
    tensor = torch.from_numpy(arr).float().unsqueeze(0)  # 1xCxHxW

    return tensor, orig_rgb, (orig_w, orig_h)


@torch.no_grad()
def predict_mask(model, tensor, orig_size):
    """Runs the model and returns a (H, W) float32 probability map resized to orig_size."""
    tensor = tensor.to(DEVICE)
    mask_pred, _, _ = model(tensor)
    prob = torch.sigmoid(mask_pred)

    orig_w, orig_h = orig_size
    prob_full = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    prob_full = prob_full.squeeze().cpu().numpy()
    return prob_full


def make_overlay(orig_rgb: np.ndarray, prob_map: np.ndarray, threshold: float,
                  overlay_color=(0, 255, 0), overlay_alpha=0.45, draw_contour=True) -> np.ndarray:
    """
    Draws a translucent colored overlay wherever prob_map > threshold,
    plus a contour outline around the detected region.
    Returns an RGB uint8 image.
    """
    binary_mask = (prob_map > threshold).astype(np.uint8)

    overlay = orig_rgb.copy()
    color_layer = np.zeros_like(orig_rgb)
    color_layer[:, :] = overlay_color
    mask_3ch = np.repeat(binary_mask[:, :, None], 3, axis=2)

    blended = (orig_rgb.astype(np.float32) * (1 - overlay_alpha) +
               color_layer.astype(np.float32) * overlay_alpha).astype(np.uint8)
    overlay = np.where(mask_3ch == 1, blended, overlay)

    if draw_contour and binary_mask.sum() > 0:
        contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)  # red contour, BGR
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    return overlay


def make_side_by_side(orig_rgb: np.ndarray, prob_map: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Original | heatmap | overlay, concatenated horizontally."""
    h, w = orig_rgb.shape[:2]

    heatmap_gray = (prob_map * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    panels = [orig_rgb, heatmap_color, overlay]
    panels = [cv2.resize(p, (w, h)) for p in panels]
    return np.concatenate(panels, axis=1)


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image inference and visualize the predicted mask.")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to the input image.")
    parser.add_argument("--ckpt", default=CKPT_PATH, help="Path to the model checkpoint (.pth).")
    parser.add_argument("--out", default=OUT_PATH,
                         help="Path to save the overlay result. Defaults to 'results/<image_name>_overlay.png'.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Probability threshold for the binary mask.")
    parser.add_argument("--side-by-side", action="store_true", default=SIDE_BY_SIDE,
                         help="Also save an original | heatmap | overlay comparison image.")
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    if not args.image or not os.path.exists(args.image):
        print(f"\nNote: Test image {args.image!r} not found on disk.")
        print("Provide an existing image path via --image /path/to/image.jpg to run inference.")
        return

    print(f"\nDevice : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model...")
    model = load_model(args.ckpt)

    print(f"Running inference on: {args.image}")
    tensor, orig_rgb, orig_size = load_and_preprocess(args.image)
    prob_map = predict_mask(model, tensor, orig_size)

    coverage = float((prob_map > args.threshold).mean())
    print(f"  Predicted foreground coverage: {coverage*100:.2f}% of image "
          f"(threshold={args.threshold})")

    overlay = make_overlay(orig_rgb, prob_map, args.threshold)

    out_path = args.out
    if out_path is None:
        results_dir = str(PROJECT_ROOT / "results")
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(results_dir, f"{filename}_overlay.png")

    Image.fromarray(overlay).save(out_path)
    print(f"  Overlay saved to: {out_path}")

    if args.side_by_side:
        combo = make_side_by_side(orig_rgb, prob_map, overlay)
        combo_path = os.path.splitext(out_path)[0] + "_sidebyside.png"
        Image.fromarray(combo).save(combo_path)
        print(f"  Side-by-side (original | heatmap | overlay) saved to: {combo_path}")

    print()


if __name__ == "__main__":
    main()
