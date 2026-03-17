"""
download_dataset.py
===================
Dataset preparation script for YOLO weed-detection training.

What this script does
---------------------
This script downloads and organises a public weed-detection dataset from
Roboflow Universe into the standard YOLOv8 directory layout:

    dataset/
    ├── images/
    │   ├── train/   (80 %)
    │   ├── val/     (10 %)
    │   └── test/    (10 %)
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml    ← YOLO training config

Recommended public datasets (free, no login required via direct links)
-----------------------------------------------------------------------
1. Roboflow Weed Detection (primary, used by default)
   https://universe.roboflow.com/brad-dwyer/oxford-pets/dataset/2
2. WeedMap dataset (DeepWeeds alternative)
   https://github.com/AlexOlsen/DeepWeeds

Quick usage
-----------
  # Option A – Download from Roboflow (requires free API key)
  python dataset/download_dataset.py --method roboflow --api-key YOUR_KEY

  # Option B – Download the DeepWeeds dataset (no API key needed)
  python dataset/download_dataset.py --method deepweeds

  # Option C – Convert your own images (place in dataset/raw_images/)
  python dataset/download_dataset.py --method local

After running, train YOLOv8 with:
  yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
"""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path
import random

# ── directory layout ────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent
SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# Class names (update after you review the dataset labels)
CLASS_NAMES = ["weed", "soil"]

DEEPWEEDS_URL = (
    "https://github.com/AlexOlsen/DeepWeeds/releases/download/1.0/"
    "deepweeds.zip"
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_dirs():
    for split in SPLITS:
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    print("[INFO] Dataset directories created.")


def _write_data_yaml(num_classes: int, class_names: list):
    yaml_path = DATASET_ROOT / "data.yaml"
    content = (
        f"path: {DATASET_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {num_classes}\n"
        f"names: {class_names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[INFO] data.yaml written to {yaml_path}")


def _split_files(image_files: list[Path]):
    """Assign image files to train / val / test splits and return a dict."""
    random.shuffle(image_files)
    n = len(image_files)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    return {
        "train": image_files[:n_train],
        "val":   image_files[n_train:n_train + n_val],
        "test":  image_files[n_train + n_val:],
    }


def _copy_to_split(image_files: list[Path], split: str, label_src_dir: Path | None):
    img_dst = DATASET_ROOT / "images" / split
    lbl_dst = DATASET_ROOT / "labels" / split
    for img_path in image_files:
        shutil.copy2(img_path, img_dst / img_path.name)
        if label_src_dir:
            lbl_path = label_src_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dst / lbl_path.name)
    print(f"[INFO] {len(image_files)} images → {split}/")


# ── method A: Roboflow ───────────────────────────────────────────────────────

def download_roboflow(api_key: str, workspace: str, project: str, version: int):
    """
    Download a dataset from Roboflow in YOLOv8 format.
    Requires:  pip install roboflow
    """
    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        sys.exit("[ERROR] roboflow package not installed.\n  Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(DATASET_ROOT))
    print(f"[INFO] Downloaded dataset to {dataset.location}")

    # Ensure data.yaml exists
    yaml_src = Path(dataset.location) / "data.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, DATASET_ROOT / "data.yaml")
    print("[DONE] Roboflow dataset ready.")


# ── method B: DeepWeeds ──────────────────────────────────────────────────────

def download_deepweeds():
    """
    Download and organise the DeepWeeds dataset.
    DeepWeeds contains 17 509 images across 9 weed species native to Australia.
    We map all weed classes to class 0 and generate YOLO-format label files.
    """
    import zipfile  # noqa: PLC0415

    zip_path = DATASET_ROOT / "deepweeds.zip"
    extract_dir = DATASET_ROOT / "_deepweeds_raw"

    print("[INFO] Downloading DeepWeeds (~1.5 GB) …")
    print("       (This may take several minutes on a slow connection)")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, 100 * downloaded // total_size)
            print(f"\r  {pct:3d}%  {downloaded // 1_048_576} MB / {total_size // 1_048_576} MB", end="")

    urllib.request.urlretrieve(DEEPWEEDS_URL, zip_path, reporthook=_progress)
    print()

    print("[INFO] Extracting archive …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    # Find all JPEG images
    all_images = sorted(extract_dir.rglob("*.jpg")) + sorted(extract_dir.rglob("*.JPG"))
    print(f"[INFO] Found {len(all_images)} images.")

    _make_dirs()
    splits = _split_files(all_images)

    # Generate placeholder label files (full-frame bounding box, class 0 = weed)
    for split, imgs in splits.items():
        for img_path in imgs:
            shutil.copy2(img_path, DATASET_ROOT / "images" / split / img_path.name)
            lbl_path = DATASET_ROOT / "labels" / split / img_path.with_suffix(".txt").name
            # YOLO label: <class> <cx> <cy> <w> <h>  (normalised)
            lbl_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        print(f"[INFO] {len(imgs)} images → {split}/")

    shutil.rmtree(extract_dir)
    _write_data_yaml(num_classes=1, class_names=["weed"])
    print("[DONE] DeepWeeds dataset ready.")


# ── method C: local images ───────────────────────────────────────────────────

def prepare_local(raw_dir: str = "dataset/raw_images"):
    """
    Split locally stored images into train / val / test sets.

    Expects images in  dataset/raw_images/  (JPEG or PNG).
    Label files with the same stem and a .txt extension should be in the
    same folder (YOLO format).  Images without a label file are still
    copied but will have no annotations.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        sys.exit(
            f"[ERROR] Raw image directory not found: {raw_path.resolve()}\n"
            "        Place your images (and .txt label files) there and re-run."
        )

    all_images = sorted(raw_path.glob("*.jpg")) + sorted(raw_path.glob("*.jpeg")) + \
                 sorted(raw_path.glob("*.png")) + sorted(raw_path.glob("*.JPG"))

    if not all_images:
        sys.exit(f"[ERROR] No images found in {raw_path.resolve()}")

    print(f"[INFO] Found {len(all_images)} images in {raw_path}")
    _make_dirs()
    splits = _split_files(all_images)
    for split, imgs in splits.items():
        _copy_to_split(imgs, split, label_src_dir=raw_path)

    _write_data_yaml(num_classes=len(CLASS_NAMES), class_names=CLASS_NAMES)
    print("[DONE] Local dataset organised.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare weed-detection dataset for YOLO training"
    )
    sub = parser.add_subparsers(dest="method", required=True)

    # Roboflow
    rf = sub.add_parser("roboflow", help="Download from Roboflow Universe")
    rf.add_argument("--api-key", required=True, help="Roboflow API key")
    rf.add_argument("--workspace", default="weed-detection-dcl3h", help="Workspace slug")
    rf.add_argument("--project", default="weed-detection-ljcvl", help="Project slug")
    rf.add_argument("--version", type=int, default=2, help="Dataset version number")

    # DeepWeeds
    sub.add_parser("deepweeds", help="Download DeepWeeds dataset (~1.5 GB)")

    # Local
    loc = sub.add_parser("local", help="Split locally stored images")
    loc.add_argument("--raw-dir", default="dataset/raw_images", help="Folder with raw images")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.method == "roboflow":
        download_roboflow(args.api_key, args.workspace, args.project, args.version)
    elif args.method == "deepweeds":
        download_deepweeds()
    elif args.method == "local":
        prepare_local(args.raw_dir)
