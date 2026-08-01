"""
Kvasir-SEG Augmentation Pipeline
==================================
Expands Kvasir images using medically justified augmentations.
Saves augmented dataset to Kvasir-SEG-aug/ (images/ and masks/).

Augmentation strategy:
  3 variants per image:
    - Horizontal flip
    - Rotation (random ±20°)
    - Colour jitter (brightness/contrast/saturation/hue)

LOCAL STRUCTURE:
  ./
  ├── Kvasir-SEG/               ← Original dataset (images/ & masks/)
  └── Kvasir-SEG-aug/           ← Created automatically (images/ & masks/)

USAGE:
  python train/augment_kvasir.py
"""

import os
import cv2
import json
import zipfile
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
KVASIR_IMAGE_DIR  = str(PROJECT_ROOT / "Kvasir-SEG" / "images")
KVASIR_MASK_DIR   = str(PROJECT_ROOT / "Kvasir-SEG" / "masks")

OUTPUT_DIR        = str(PROJECT_ROOT / "Kvasir-SEG-aug")
OUTPUT_IMAGE_DIR  = os.path.join(OUTPUT_DIR, "images")
OUTPUT_MASK_DIR   = os.path.join(OUTPUT_DIR, "masks")

ZIP_PATH          = str(PROJECT_ROOT / "Kvasir-SEG-aug.zip")

# Augmentation parameters
ROTATION_LIMIT        = 20       # degrees
BRIGHTNESS_LIMIT      = 0.2      # ±20%
CONTRAST_LIMIT        = 0.2      # ±20%
SATURATION_LIMIT      = 20       # ±20 hue/saturation
BLUR_LIMIT            = 3
ELASTIC_ALPHA         = 30
ELASTIC_SIGMA         = 5

VARIANTS_PER_IMAGE = 2

SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────
# INSTALL CHECK
# ─────────────────────────────────────────────────────
try:
    import albumentations as A
    print("albumentations found.")
except ImportError:
    print("Installing albumentations...")
    os.system("pip install albumentations -q")
    import albumentations as A


# ─────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# ─────────────────────────────────────────────────────

def get_augmentation_pipelines():
    hflip = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])

    rotate = A.Compose([
        A.Rotate(limit=ROTATION_LIMIT, p=1.0, border_mode=cv2.BORDER_REFLECT),
    ])

    colour = A.Compose([
        A.ColorJitter(
            brightness=BRIGHTNESS_LIMIT,
            contrast=CONTRAST_LIMIT,
            saturation=SATURATION_LIMIT / 100,
            hue=0.05,
            p=1.0
        ),
    ])

    return [
        ("hflip",  hflip),
        ("rotate", rotate),
        ("colour", colour),
    ]


# ─────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────

def find_mask_path(mask_dir: str, stem: str) -> str:
    for ext in [".jpg", ".png"]:
        p = os.path.join(mask_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def build_records(image_dir: str, mask_dir: str) -> list:
    records = []
    supported = {".jpg", ".jpeg", ".png"}
    for fname in sorted(os.listdir(image_dir)):
        if Path(fname).suffix.lower() not in supported:
            continue
        stem      = Path(fname).stem
        mask_path = find_mask_path(mask_dir, stem)
        if mask_path is None:
            print(f"  Warning: no mask found for {fname}, skipping.")
            continue
        records.append({
            "img":  os.path.join(image_dir, fname),
            "mask": mask_path,
            "stem": stem,
            "ext":  Path(fname).suffix.lower(),
        })
    return records


def load_image_mask(img_path: str, mask_path: str):
    img  = cv2.imread(img_path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return img, mask


def save_image_mask(img: np.ndarray, mask: np.ndarray,
                    out_img_path: str, out_mask_path: str):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_img_path,  img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(out_mask_path, mask)


# ─────────────────────────────────────────────────────
# MAIN AUGMENTATION LOOP
# ─────────────────────────────────────────────────────

def augment_dataset():
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASK_DIR,  exist_ok=True)

    records   = build_records(KVASIR_IMAGE_DIR, KVASIR_MASK_DIR)
    pipelines = get_augmentation_pipelines()

    print(f"\nOriginal images    : {len(records)}")
    print(f"Variants per image : {len(pipelines)} + 1 original")
    print(f"Expected total     : {len(records) * (len(pipelines) + 1)}")
    print(f"Output dir         : {OUTPUT_DIR}\n")

    stats = {
        "original": 0,
        "augmented": 0,
        "failed": 0,
        "per_variant": {name: 0 for name, _ in pipelines},
    }

    for record in tqdm(records, desc="Augmenting"):
        stem = record["stem"]

        try:
            img, mask = load_image_mask(record["img"], record["mask"])
        except Exception as e:
            print(f"  Error loading {stem}: {e}")
            stats["failed"] += 1
            continue

        orig_img_path  = os.path.join(OUTPUT_IMAGE_DIR, f"{stem}.jpg")
        orig_mask_path = os.path.join(OUTPUT_MASK_DIR,  f"{stem}.png")
        save_image_mask(img, mask, orig_img_path, orig_mask_path)
        stats["original"] += 1

        for variant_name, pipeline in pipelines:
            try:
                augmented = pipeline(image=img, mask=mask)
                aug_img   = augmented["image"]
                aug_mask  = augmented["mask"]

                out_stem      = f"{stem}_{variant_name}"
                out_img_path  = os.path.join(OUTPUT_IMAGE_DIR, f"{out_stem}.jpg")
                out_mask_path = os.path.join(OUTPUT_MASK_DIR,  f"{out_stem}.png")

                save_image_mask(aug_img, aug_mask, out_img_path, out_mask_path)
                stats["augmented"] += 1
                stats["per_variant"][variant_name] += 1

            except Exception as e:
                print(f"  Warning: {variant_name} failed for {stem}: {e}")
                stats["failed"] += 1

    return stats


# ─────────────────────────────────────────────────────
# VERIFY OUTPUT
# ─────────────────────────────────────────────────────

def verify_output():
    images = list(Path(OUTPUT_IMAGE_DIR).glob("*.jpg"))
    masks  = list(Path(OUTPUT_MASK_DIR).glob("*.png"))

    print(f"\n  Images in output : {len(images)}")
    print(f"  Masks in output  : {len(masks)}")

    missing = 0
    for img_path in images:
        stem      = img_path.stem
        mask_path = Path(OUTPUT_MASK_DIR) / f"{stem}.png"
        if not mask_path.exists():
            print(f"  Missing mask for: {stem}")
            missing += 1

    if missing == 0:
        print(f"  ✓ All images have corresponding masks.")
    else:
        print(f"  ✗ {missing} images missing masks.")

    if len(images) > 0:
        sample_img  = images[0]
        sample_mask = Path(OUTPUT_MASK_DIR) / f"{sample_img.stem}.png"
        if sample_mask.exists():
            img  = cv2.imread(str(sample_img))
            mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
            print(f"\n  Sample check:")
            print(f"    Image : {sample_img.name}  shape={img.shape}")
            print(f"    Mask  : {sample_mask.name} shape={mask.shape}")

    return len(images), len(masks)


def create_local_zip():
    print(f"\n  Zipping augmented dataset to {ZIP_PATH}...")

    all_files = (
        list(Path(OUTPUT_IMAGE_DIR).glob("*.jpg")) +
        list(Path(OUTPUT_MASK_DIR).glob("*.png"))
    )

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, fpath in enumerate(all_files):
            arcname = os.path.relpath(str(fpath), OUTPUT_DIR)
            zf.write(str(fpath), arcname)

    zip_size = os.path.getsize(ZIP_PATH) / (1024**3)
    print(f"\n  ✓ Zip saved: {ZIP_PATH} ({zip_size:.2f} GB)")


def save_manifest(stats: dict, n_images: int, n_masks: int):
    manifest = {
        "source":              KVASIR_IMAGE_DIR,
        "output":              OUTPUT_DIR,
        "original_images":     stats["original"],
        "augmented_images":    stats["augmented"],
        "total_images":        n_images,
        "total_masks":         n_masks,
        "failed":              stats["failed"],
        "augmentation_factor": round(n_images / max(stats["original"], 1), 1),
        "variants_applied":    stats["per_variant"],
        "parameters": {
            "rotation_limit":    ROTATION_LIMIT,
            "brightness_limit":  BRIGHTNESS_LIMIT,
            "contrast_limit":    CONTRAST_LIMIT,
            "saturation_limit":  SATURATION_LIMIT,
            "blur_limit":        BLUR_LIMIT,
            "elastic_alpha":     ELASTIC_ALPHA,
            "elastic_sigma":     ELASTIC_SIGMA,
            "seed":              SEED,
        },
    }

    manifest_path = os.path.join(OUTPUT_DIR, "augmentation_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Manifest saved to: {manifest_path}")
    return manifest


def main():
    if not os.path.exists(KVASIR_IMAGE_DIR):
        print(f"\nERROR: Kvasir images not found at {KVASIR_IMAGE_DIR}")
        return

    if os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_IMAGE_DIR)) > 2500:
        print(f"\nAugmented dataset already exists at {OUTPUT_DIR}")
        print(f"Images found: {len(os.listdir(OUTPUT_IMAGE_DIR))}")
        return

    print(f"\n{'='*60}")
    print(f"  Kvasir-SEG Augmentation Pipeline")
    print(f"  Expands images 3x (hflip, rotate, colour jitter)")
    print(f"{'='*60}\n")

    print("Step 1/3 — Generating augmented images...")
    stats = augment_dataset()

    print(f"\nStep 2/3 — Verifying output...")
    n_images, n_masks = verify_output()

    print(f"\nStep 3/3 — Saving manifest...")
    manifest = save_manifest(stats, n_images, n_masks)
    create_local_zip()

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION COMPLETE")
    print(f"  Total image-mask pairs : {n_images}")
    print(f"  Output directory       : {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
