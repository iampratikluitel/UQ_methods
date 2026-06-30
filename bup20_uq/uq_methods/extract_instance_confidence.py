"""
extract_instance_confidences.py
Matches predicted instances to ground truth via IoU to get per-instance
(confidence, correct/incorrect, class, area, position) for calibration analysis.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from config import DEFAULT_SCORE_THRESH, DEFAULT_WEIGHTS, OUTPUT_DIR, resolve_image_path


def build_predictor(weights_path, score_thresh=0.3):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def box_iou(box1, box2):
    """box format: [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_predictions_to_gt(pred_boxes, pred_scores, pred_classes,
                             gt_boxes, gt_classes, iou_thresh=0.5):
    """
    For each prediction, find best-matching GT box.
    Returns list of dicts: {score, predicted_class, correct, matched_gt_idx, box}
    """
    results = []
    matched_gt = set()

    # Sort predictions by score descending (greedy matching)
    order = np.argsort(-pred_scores)

    for idx in order:
        box = pred_boxes[idx]
        score = pred_scores[idx]
        pred_cls = pred_classes[idx]

        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        is_correct = False
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            is_correct = (gt_classes[best_gt_idx] == pred_cls)
            matched_gt.add(best_gt_idx)

        # Position in image — normalized centroid
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        results.append({
            "score": float(score),
            "predicted_class": int(pred_cls),
            "correct": bool(is_correct),
            "iou_with_gt": float(best_iou),
            "box": [float(b) for b in box],
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "area": float((box[2]-box[0]) * (box[3]-box[1])),
        })

    return results


def extract_all_instance_data(dataset_name, weights_path, score_thresh=0.3,
                                output_path="instance_calibration_data.json"):
    predictor = build_predictor(weights_path, score_thresh)
    dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    class_names = metadata.thing_classes

    all_instances = []

    for sample in tqdm(dataset, desc="Extracting instance data"):
        import cv2
        image_path = resolve_image_path(sample["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]

        outputs = predictor(image)
        instances = outputs["instances"]

        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_scores = instances.scores.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()

        # Ground truth boxes from annotations (XYXY format)
        gt_boxes = []
        gt_classes = []
        for ann in sample["annotations"]:
            bbox = ann["bbox"]
            if ann.get("bbox_mode", BoxMode.XYWH_ABS) == BoxMode.XYWH_ABS:
                x, y, w, h = bbox
                gt_boxes.append([x, y, x+w, y+h])
            else:
                gt_boxes.append(bbox)
            gt_classes.append(ann["category_id"])

        if len(pred_boxes) == 0:
            continue

        matched = match_predictions_to_gt(
            pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes
        )

        for m in matched:
            m["image_id"] = sample["image_id"]
            m["file_name"] = image_path
            m["img_width"] = img_w
            m["img_height"] = img_h
            m["norm_x"] = m["centroid_x"] / img_w   # 0=left, 1=right
            m["norm_y"] = m["centroid_y"] / img_h   # 0=top, 1=bottom
            m["class_name"] = class_names[m["predicted_class"]]
            all_instances.append(m)

    with open(output_path, "w") as f:
        json.dump(all_instances, f, indent=2)

    print(f"\nExtracted {len(all_instances)} instances → {output_path}")
    return all_instances


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pepper_eval")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thresh", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "instance_calibration_data.json"))
    args = parser.parse_args()

    extract_all_instance_data(args.dataset, args.weights, args.thresh, args.output)