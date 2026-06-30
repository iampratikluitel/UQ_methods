"""
Visual comparison: GT vs baseline / TTA / MC Dropout on original images,
plus per-TTA-transform panels for each selected eval image.

Output layout:
  output/method_comparison/
    image_{id}/
      comparison_all_methods.png   # GT | Baseline | TTA | MC Dropout
      tta/
        original.png               # GT | Prediction
        hflip.png                  # Augmented input | Prediction
        ...
"""

from __future__ import annotations

import argparse
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as detection_utils
from detectron2.utils.visualizer import Visualizer

from config import (
    DEFAULT_DATASET,
    DEFAULT_SCORE_THRESH,
    DEFAULT_WEIGHTS,
    METHOD_COMPARISON_DIR,
    resolve_image_path,
)
from extract_instance_uq import build_mc_dropout_model
from tta import get_augmentations


def build_predictor(weights_path, score_thresh=DEFAULT_SCORE_THRESH):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def mc_dropout_predict(image_bgr, model, cfg):
    """Single stochastic MC Dropout forward pass on a BGR image."""
    from detectron2.data import transforms as T

    h, w = image_bgr.shape[:2]
    image_rgb = image_bgr[:, :, ::-1]
    aug = T.ResizeShortestEdge(short_edge_length=[800], max_size=1333, sample_style="choice")
    input_image = aug.get_transform(image_rgb).apply_image(image_rgb)
    input_tensor = torch.as_tensor(input_image.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": input_tensor, "height": h, "width": w}]
    with torch.no_grad():
        outputs = model(inputs)
    return outputs[0]["instances"].to("cpu")


def render_gt(image_bgr, sample, metadata):
    vis = Visualizer(image_bgr[:, :, ::-1], metadata=metadata)
    gt = detection_utils.annotations_to_instances(
        sample["annotations"], image_bgr.shape[:2]
    )
    masks = gt.gt_masks if gt.has("gt_masks") else None
    labels = [metadata.thing_classes[c] for c in gt.gt_classes]
    return vis.overlay_instances(boxes=gt.gt_boxes, labels=labels, masks=masks).get_image()


def render_pred(image_bgr, instances, metadata):
    vis = Visualizer(image_bgr[:, :, ::-1], metadata=metadata)
    return vis.draw_instance_predictions(instances).get_image()


def save_panels(panels, titles, save_path, figsize_per_col=5):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col * n, figsize_per_col * 1.6))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_all_methods_comparison(
    image_bgr, sample, metadata, predictor, mc_model, mc_cfg, save_path
):
    gt_img = render_gt(image_bgr, sample, metadata)
    baseline_inst = predictor(image_bgr)["instances"].to("cpu")
    tta_inst = baseline_inst  # TTA prediction on original image
    mc_inst = mc_dropout_predict(image_bgr, mc_model, mc_cfg)

    panels = [
        gt_img,
        render_pred(image_bgr, baseline_inst, metadata),
        render_pred(image_bgr, tta_inst, metadata),
        render_pred(image_bgr, mc_inst, metadata),
    ]
    save_panels(
        panels,
        ["Ground Truth", "Baseline", "TTA (original)", "MC Dropout"],
        save_path,
        figsize_per_col=4.5,
    )


def save_tta_transform_panels(
    image_bgr, sample, metadata, predictor, tta_dir, show_gt_on_original=True
):
    tta_dir.mkdir(parents=True, exist_ok=True)

    for name, aug_fn, _ in get_augmentations():
        aug_img = aug_fn(image_bgr)
        instances = predictor(aug_img)["instances"].to("cpu")
        pred_img = render_pred(aug_img, instances, metadata)

        if name == "original" and show_gt_on_original:
            gt_img = render_gt(image_bgr, sample, metadata)
            save_panels(
                [gt_img, pred_img],
                ["Ground Truth", f"Prediction ({name})"],
                tta_dir / f"{name}.png",
            )
        else:
            input_rgb = aug_img[:, :, ::-1]
            save_panels(
                [input_rgb, pred_img],
                [f"Input ({name})", f"Prediction ({name})"],
                tta_dir / f"{name}.png",
            )


def process_sample(
    sample, metadata, predictor, mc_model, mc_cfg, out_root,
):
    sample = dict(sample)
    image_path = resolve_image_path(sample["file_name"])
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"  Skip missing image: {image_path}")
        return

    image_id = sample["image_id"]
    sample_dir = out_root / f"image_{image_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_all_methods_comparison(
        image_bgr,
        sample,
        metadata,
        predictor,
        mc_model,
        mc_cfg,
        sample_dir / "comparison_all_methods.png",
    )
    save_tta_transform_panels(
        image_bgr,
        sample,
        metadata,
        predictor,
        sample_dir / "tta",
    )


def run_method_comparison(
    dataset_name=DEFAULT_DATASET,
    weights_path=DEFAULT_WEIGHTS,
    score_thresh=DEFAULT_SCORE_THRESH,
    dropout_rate=0.3,
    out_dir=None,
    n_samples=5,
    image_ids=None,
    seed=42,
):
    out_root = Path(out_dir or METHOD_COMPARISON_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(weights_path, score_thresh)
    mc_model, mc_cfg = build_mc_dropout_model(weights_path, score_thresh, dropout_rate)
    mc_cfg.MODEL.WEIGHTS = str(weights_path)

    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    if image_ids:
        id_set = set(image_ids)
        samples = [s for s in dataset if s["image_id"] in id_set]
    else:
        rng = random.Random(seed)
        samples = rng.sample(dataset, min(n_samples, len(dataset)))

    print(f"\nMethod comparison → {out_root}")
    print(f"  Dataset: {dataset_name}, images: {len(samples)}")

    for sample in tqdm(samples, desc="Method comparison"):
        process_sample(sample, metadata, predictor, mc_model, mc_cfg, out_root)

    print(f"\nDone. See {out_root}/image_*/")
    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT vs baseline/TTA/MC Dropout visual comparison")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thresh", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--out", default=str(METHOD_COMPARISON_DIR))
    parser.add_argument("--n", type=int, default=5, help="Random samples if --image-id not set")
    parser.add_argument("--image-id", type=int, nargs="*", help="Specific image IDs to visualize")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_method_comparison(
        dataset_name=args.dataset,
        weights_path=Path(args.weights),
        score_thresh=args.thresh,
        dropout_rate=args.dropout,
        out_dir=args.out,
        n_samples=args.n,
        image_ids=args.image_id,
        seed=args.seed,
    )
