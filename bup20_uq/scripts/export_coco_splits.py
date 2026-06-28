#!/usr/bin/env python3
"""Step 0: Export BUP20 train/val/eval COCO JSON for Bender."""

import json
import os
from copy import deepcopy
from pathlib import Path

import yaml

# --- CONFIG (edit if your paths differ) ---
DATA_ROOT = Path("/home/s29pluit/datasets/bup_20/CKA_sweet_pepper_2020_summer")
OUT_DIR = Path("/home/s29pluit/projects/bup20_uq/data")

# Skip keypoint-only class; keep ripeness classes
SKIP_CATEGORY_IDS = {11}  # pepper_kp
VALID_CATEGORY_IDS = {12, 13, 14, 17, 18}  # red, yellow, green, mixed_red, mixed_yellow

SPLIT_MAP = {
    "train": "train",
    "val": "valid",   # yaml uses "valid", not "val"
    "eval": "eval",
}


def resolve_image_path(img: dict) -> str:
    """Build absolute path on Bender from JSON metadata."""
    old_path = img["path"]  # e.g. /datasets/CKA_sweet_pepper_2020_summer/20200924/row2/file.tiff
    rel = old_path.split("CKA_sweet_pepper_2020_summer/", 1)[-1]
    return str(DATA_ROOT / rel)


def export_split(name: str, yaml_key: str, raw: dict, image_ids: set, out_dir: Path):
    images = []
    for img in raw["images"]:
        if not img.get("annotated", False):
            continue
        if img["id"] not in image_ids:
            continue

        new_img = deepcopy(img)
        abs_path = resolve_image_path(img)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Missing image: {abs_path}")

        new_img["file_name"] = abs_path  # full path works with Detectron2
        images.append(new_img)

    image_id_set = {img["id"] for img in images}

    annotations = []
    for ann in raw["annotations"]:
        if ann["image_id"] not in image_id_set:
            continue
        if ann["category_id"] in SKIP_CATEGORY_IDS:
            continue
        if not ann.get("segmentation"):
            continue
        annotations.append(ann)

    categories = [c for c in raw["categories"] if c["id"] in VALID_CATEGORY_IDS]

    out = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_path = out_dir / f"annotations_{name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"{name:5s} -> {out_path}")
    print(f"       images={len(images)}, annotations={len(annotations)}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_ROOT / "CKA_sweet_pepper_2020_summer.json") as f:
        raw = json.load(f)

    with open(DATA_ROOT / "CKA_sweet_pepper_2020_summer.yaml") as f:
        cfg = yaml.safe_load(f)

    for split_name, yaml_key in SPLIT_MAP.items():
        image_ids = set(cfg["image_sets"][yaml_key])
        export_split(split_name, yaml_key, raw, image_ids, OUT_DIR)

    print("\nStep 0 export done.")


if __name__ == "__main__":
    main()
