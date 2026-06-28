from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import json

DATA_DIR = "/home/s29pluit/projects/bup20_uq/data"
IMAGE_ROOT = "/home/s29pluit/datasets/bup_20"

# Pull class names directly from your JSON
with open(f"{DATA_DIR}/annotations_train.json") as f:
    coco = json.load(f)

CLASSES = [cat["name"] for cat in sorted(coco["categories"], key=lambda x: x["id"])]
print("Classes found:", CLASSES)

# Register all three splits
for split, json_file in [
    ("pepper_train", "annotations_train.json"),
    ("pepper_val",   "annotations_val.json"),
    ("pepper_eval",  "annotations_eval.json"),
]:
    register_coco_instances(
        split,
        {"thing_classes": CLASSES},
        f"{DATA_DIR}/{json_file}",
        IMAGE_ROOT,
    )
    MetadataCatalog.get(split).set(thing_classes=CLASSES)

print("Datasets registered successfully!")