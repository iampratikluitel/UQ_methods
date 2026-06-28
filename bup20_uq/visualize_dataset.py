import random
import cv2
import matplotlib.pyplot as plt

import register_dataset

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

dataset_dicts = DatasetCatalog.get("pepper_train")
metadata = MetadataCatalog.get("pepper_train")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])

    vis = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.7
    )

    out = vis.draw_dataset_dict(d)

    plt.figure(figsize=(8,12))
    plt.imshow(out.get_image())
    plt.axis("off")
    plt.show()

    plt.savefig(f"vis_{d['image_id']}.png", bbox_inches="tight", dpi=150)
    plt.close()