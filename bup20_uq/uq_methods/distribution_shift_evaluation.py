"""
distribution_shift_evaluation.py
Applies controlled rotation as distribution shift and tracks confidence/detection drift.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog

from config import DEFAULT_SCORE_THRESH, DEFAULT_WEIGHTS, OUTPUT_DIR, resolve_image_path
from plotting import plot_shift_robustness


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def build_predictor(weights_path, score_thresh=0.3):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def evaluate_under_shift(
    dataset_name,
    weights_path,
    shift_angles=(0, 15, 30, 45, 60, 90),
    score_thresh=DEFAULT_SCORE_THRESH,
):
    predictor = build_predictor(weights_path, score_thresh)
    dataset = DatasetCatalog.get(dataset_name)
    shift_results = {}

    for angle in shift_angles:
        all_scores = []
        all_n_dets = []

        for sample in tqdm(dataset, desc=f"Rotation {angle}°"):
            image_path = resolve_image_path(sample["file_name"])
            image = cv2.imread(image_path)
            if image is None:
                continue
            shifted = rotate_image(image, angle) if angle != 0 else image

            outputs = predictor(shifted)
            scores = outputs["instances"].scores.cpu().numpy()
            all_scores.append(scores.mean() if len(scores) > 0 else 0.0)
            all_n_dets.append(len(scores))

        shift_results[str(angle)] = {
            "mean_confidence": float(np.mean(all_scores)),
            "std_confidence": float(np.std(all_scores)),
            "mean_detections": float(np.mean(all_n_dets)),
        }
        print(f"  Angle {angle}°: mean_conf={shift_results[str(angle)]['mean_confidence']:.3f}, "
              f"mean_dets={shift_results[str(angle)]['mean_detections']:.2f}")

    return shift_results


def run_distribution_shift(
    dataset_name: str,
    weights_path: Path,
    out_dir: Path,
    shift_angles=(0, 15, 30, 45, 60, 90),
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_under_shift(dataset_name, weights_path, shift_angles)

    json_path = out_dir / "distribution_shift_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_shift_robustness(
        results,
        save_path=str(out_dir / "shift_robustness_curve.png"),
    )
    print(f"Shift results saved to {json_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="pepper_eval")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out", default=str(OUTPUT_DIR / "uq_report"))
    args = parser.parse_args()

    run_distribution_shift(args.dataset, Path(args.weights), Path(args.out))
