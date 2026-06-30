"""
Per-instance calibration data for TTA and MC Dropout.

Produces the same JSON schema as extract_instance_confidence.py so
run_calibration_analysis can compute ECE, ACE, and confidence gap.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.modeling import build_model
from detectron2.structures import BoxMode

from config import DEFAULT_SCORE_THRESH, DEFAULT_WEIGHTS, resolve_image_path
from extract_instance_confidence import box_iou, match_predictions_to_gt
from mc_dropout import build_cfg as build_mc_cfg, inject_dropout_roi_heads, preprocess_image
from tta import get_augmentations

IOU_MATCH_THRESH = 0.5

# Spatial-preserving augmentations for per-instance TTA calibration.
CALIBRATION_AUG_NAMES = {
    "original", "hflip", "vflip", "brightness+", "brightness-",
}


def parse_gt_from_sample(sample):
    gt_boxes, gt_classes = [], []
    for ann in sample["annotations"]:
        bbox = ann["bbox"]
        if ann.get("bbox_mode", BoxMode.XYWH_ABS) == BoxMode.XYWH_ABS:
            x, y, w, h = bbox
            gt_boxes.append([x, y, x + w, y + h])
        else:
            gt_boxes.append(bbox)
        gt_classes.append(ann["category_id"])
    return gt_boxes, gt_classes


def instances_from_predictor_output(output):
    inst = output["instances"]
    return {
        "boxes": inst.pred_boxes.tensor.cpu().numpy(),
        "scores": inst.scores.cpu().numpy(),
        "classes": inst.pred_classes.cpu().numpy(),
    }


def transform_boxes_to_original(boxes, img_w, img_h, aug_name):
    """Map augmented-image boxes back to original image coordinates."""
    if len(boxes) == 0:
        return boxes
    out = []
    for x1, y1, x2, y2 in boxes:
        if aug_name == "hflip":
            out.append([img_w - x2, y1, img_w - x1, y2])
        elif aug_name == "vflip":
            out.append([x1, img_h - y2, x2, img_h - y1])
        else:
            out.append([x1, y1, x2, y2])
    return np.array(out, dtype=float)


def cluster_scores_across_passes(anchor_boxes, anchor_classes, passes, iou_thresh=IOU_MATCH_THRESH):
    """
    Anchor detections on pass 0; match same-class boxes in other passes by IoU.
    Returns list of {box, predicted_class, score, uncertainty, n_matches}.
    """
    if len(anchor_boxes) == 0:
        return []

    clusters = []
    for i in range(len(anchor_boxes)):
        matched_scores = [float(passes[0]["scores"][i])]
        for p in passes[1:]:
            best_iou, best_j = 0.0, -1
            for j in range(len(p["boxes"])):
                if int(anchor_classes[i]) != int(p["classes"][j]):
                    continue
                iou = box_iou(anchor_boxes[i], p["boxes"][j])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_thresh:
                matched_scores.append(float(p["scores"][best_j]))

        clusters.append({
            "box": anchor_boxes[i],
            "predicted_class": int(anchor_classes[i]),
            "score": float(np.mean(matched_scores)),
            "uncertainty": float(np.std(matched_scores)) if len(matched_scores) > 1 else 0.0,
            "n_matches": len(matched_scores),
        })
    return clusters


def finalize_instance_records(clusters, gt_boxes, gt_classes, sample, image_path,
                              class_names, img_h, img_w):
    """Match clustered predictions to GT and attach metadata."""
    if not clusters:
        return []

    boxes = np.array([c["box"] for c in clusters])
    scores = np.array([c["score"] for c in clusters])
    classes = np.array([c["predicted_class"] for c in clusters])

    matched = match_predictions_to_gt(
        boxes, scores, classes, gt_boxes, gt_classes, iou_thresh=IOU_MATCH_THRESH
    )

    records = []
    for m, c in zip(matched, clusters):
        m["uncertainty"] = c["uncertainty"]
        m["n_uq_matches"] = c["n_matches"]
        m["image_id"] = sample["image_id"]
        m["file_name"] = image_path
        m["img_width"] = img_w
        m["img_height"] = img_h
        m["norm_x"] = m["centroid_x"] / img_w
        m["norm_y"] = m["centroid_y"] / img_h
        m["class_name"] = class_names[m["predicted_class"]]
        records.append(m)
    return records


def get_calibration_augmentations():
    return [a for a in get_augmentations() if a[0] in CALIBRATION_AUG_NAMES]


def cluster_tta_instances(predictor, image, img_w, img_h):
    """Per-instance TTA using spatial-preserving augmentations only."""
    augmentations = get_calibration_augmentations()
    passes = []

    for name, aug_fn, _ in augmentations:
        aug_img = aug_fn(image)
        with torch.no_grad():
            out = predictor(aug_img)
        inst = instances_from_predictor_output(out)
        inst["boxes"] = transform_boxes_to_original(inst["boxes"], img_w, img_h, name)
        passes.append(inst)

    anchor = passes[0]
    return cluster_scores_across_passes(
        anchor["boxes"], anchor["classes"], passes, iou_thresh=IOU_MATCH_THRESH
    )


def mc_dropout_passes(image_path, model, cfg, T):
    """Collect instance lists from T stochastic forward passes."""
    input_tensor, orig_size = preprocess_image(image_path, cfg)
    inputs = [{
        "image": input_tensor,
        "height": orig_size[0],
        "width": orig_size[1],
    }]

    passes = []
    with torch.no_grad():
        for _ in range(T):
            outputs = model(inputs)
            passes.append(instances_from_predictor_output(outputs[0]))
    return passes


def build_mc_dropout_model(weights_path, score_thresh, dropout_rate):
    cfg = build_mc_cfg(score_thresh=score_thresh)
    cfg.MODEL.WEIGHTS = str(weights_path)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(weights_path))
    model = inject_dropout_roi_heads(model, dropout_rate=dropout_rate)
    model.eval()
    return model, cfg


def extract_tta_instance_data(
    dataset_name,
    weights_path,
    score_thresh=DEFAULT_SCORE_THRESH,
    output_path="instance_calibration_data.json",
):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)

    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes
    all_instances = []

    aug_names = [a[0] for a in get_calibration_augmentations()]
    print(f"TTA instance calibration augmentations: {aug_names}")

    for sample in tqdm(dataset, desc="TTA instance extraction"):
        image_path = resolve_image_path(sample["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        gt_boxes, gt_classes = parse_gt_from_sample(sample)
        clusters = cluster_tta_instances(predictor, image, img_w, img_h)
        if not clusters:
            continue

        records = finalize_instance_records(
            clusters, gt_boxes, gt_classes, sample, image_path,
            class_names, img_h, img_w,
        )
        all_instances.extend(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_instances, f, indent=2)

    print(f"\nTTA: extracted {len(all_instances)} instances → {output_path}")
    return all_instances


def extract_mc_dropout_instance_data(
    dataset_name,
    weights_path,
    score_thresh=DEFAULT_SCORE_THRESH,
    T=30,
    dropout_rate=0.3,
    output_path="instance_calibration_data.json",
):
    model, cfg = build_mc_dropout_model(weights_path, score_thresh, dropout_rate)
    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes
    all_instances = []

    print(f"\nMC Dropout instance extraction: T={T}, dropout={dropout_rate}")

    for sample in tqdm(dataset, desc="MC Dropout instance extraction"):
        image_path = resolve_image_path(sample["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        gt_boxes, gt_classes = parse_gt_from_sample(sample)
        passes = mc_dropout_passes(image_path, model, cfg, T=T)
        if len(passes[0]["boxes"]) == 0:
            continue

        clusters = cluster_scores_across_passes(
            passes[0]["boxes"], passes[0]["classes"], passes,
        )
        records = finalize_instance_records(
            clusters, gt_boxes, gt_classes, sample, image_path,
            class_names, img_h, img_w,
        )
        all_instances.extend(records)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_instances, f, indent=2)

    print(f"\nMC Dropout: extracted {len(all_instances)} instances → {output_path}")
    return all_instances


if __name__ == "__main__":
    import argparse
    from config import OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["tta", "mc_dropout"], required=True)
    parser.add_argument("--dataset", default="pepper_eval")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thresh", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    out = args.output or str(
        OUTPUT_DIR / "uq_report" / args.method / "instance_calibration_data.json"
    )

    if args.method == "tta":
        extract_tta_instance_data(args.dataset, args.weights, args.thresh, out)
    else:
        extract_mc_dropout_instance_data(
            args.dataset, args.weights, args.thresh, args.T, args.dropout, out
        )
