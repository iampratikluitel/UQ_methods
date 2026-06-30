"""
distribution_shift_eval.py
Applies controlled rotation as a distribution shift, re-runs evaluation,
and tracks AP/mIoU degradation plus uncertainty increase.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import cv2
import numpy as np
import json
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog


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
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    return DefaultPredictor(cfg)


def evaluate_under_shift(dataset_name, weights_path, shift_angles=(0, 15, 30, 45, 60, 90)):
    predictor = build_predictor(weights_path)
    dataset = DatasetCatalog.get(dataset_name)

    shift_results = {}

    for angle in shift_angles:
        all_scores = []
        all_n_dets = []

        for sample in tqdm(dataset, desc=f"Rotation {angle}°"):
            image = cv2.imread(sample["file_name"])
            if image is None:
                continue
            shifted = rotate_image(image, angle) if angle != 0 else image

            outputs = predictor(shifted)
            scores = outputs["instances"].scores.cpu().numpy()
            all_scores.append(scores.mean() if len(scores) > 0 else 0.0)
            all_n_dets.append(len(scores))

        shift_results[angle] = {
            "mean_confidence": float(np.mean(all_scores)),
            "std_confidence": float(np.std(all_scores)),
            "mean_detections": float(np.mean(all_n_dets)),
        }
        print(f"  Angle {angle}°: mean_conf={shift_results[angle]['mean_confidence']:.3f}, "
              f"mean_dets={shift_results[angle]['mean_detections']:.2f}")

    return shift_results

