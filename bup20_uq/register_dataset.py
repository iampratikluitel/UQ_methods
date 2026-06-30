from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("BUP20_DATA_DIR", PROJECT_ROOT / "data"))
IMAGE_ROOT = os.environ.get(
    "BUP20_IMAGE_ROOT",
    "/home/s29pluit/datasets/bup_20",
)

with open(DATA_DIR / "annotations_train.json") as f:
    coco = json.load(f)

CLASSES = [cat["name"] for cat in sorted(coco["categories"], key=lambda x: x["id"])]
print("Classes found:", CLASSES)

for split, json_file in [
    ("pepper_train", "annotations_train.json"),
    ("pepper_val", "annotations_val.json"),
    ("pepper_eval", "annotations_eval.json"),
]:
    register_coco_instances(
        split,
        {"thing_classes": CLASSES},
        str(DATA_DIR / json_file),
        IMAGE_ROOT,
    )
    MetadataCatalog.get(split).set(thing_classes=CLASSES)

print("Datasets registered successfully!")
