

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import torch
import numpy as np
import cv2
import json
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# ── Config ────────────────────────────────────────────────────────────────────

def build_cfg(score_thresh=0.0):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = "/home/s29pluit/projects/bup20_uq/output/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.OUTPUT_DIR = "/home/s29pluit/projects/bup20_uq/output/tta"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# ── Augmentation functions ────────────────────────────────────────────────────

def get_augmentations():
    """
    Returns list of (name, forward_fn, inverse_fn) tuples.
    Inverse is needed to map predictions back to original image space.
    For score-based UQ we only need forward.
    """
    return [
        ("original",    lambda img: img,                                      lambda img: img),
        ("hflip",       lambda img: cv2.flip(img, 1),                         lambda img: cv2.flip(img, 1)),
        ("vflip",       lambda img: cv2.flip(img, 0),                         lambda img: cv2.flip(img, 0)),
        ("rotate90cw",  lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("rotate90ccw", lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("brightness+", lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=30),  lambda img: img),
        ("brightness-", lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=-30), lambda img: img),
    ]


# ── Per-image TTA ─────────────────────────────────────────────────────────────

def tta_single_image(image_path, predictor, augmentations):
    """
    Run all augmentations on one image.
    Returns:
        mean_scores     : np.ndarray of mean confidence per detection
        std_scores      : np.ndarray of std (uncertainty) per detection
        orig_output     : raw Detectron2 output on original image
        all_score_lists : list of score arrays from each augmentation
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    all_score_lists = []

    for name, aug_fn, _ in augmentations:
        augmented = aug_fn(image)
        with torch.no_grad():
            output = predictor(augmented)
        scores = output["instances"].scores.cpu().numpy()
        all_score_lists.append(scores)

    # Original image output (used as the actual prediction)
    orig_output = predictor(image)

    # Aggregate uncertainty: mean and std of score counts across augmentations
    # We use number of detections and mean score as a proxy
    detection_counts = np.array([len(s) for s in all_score_lists], dtype=float)
    mean_detections  = detection_counts.mean()
    std_detections   = detection_counts.std()

    mean_scores_per_aug = np.array([
        s.mean() if len(s) > 0 else 0.0 for s in all_score_lists
    ])
    mean_score   = mean_scores_per_aug.mean()
    std_score    = mean_scores_per_aug.std()  # ← TTA uncertainty signal

    return {
        "mean_score":       mean_score,
        "std_score":        std_score,        # higher = more uncertain
        "mean_detections":  mean_detections,
        "std_detections":   std_detections,   # high variance = unstable detections
        "orig_output":      orig_output,
        "all_score_lists":  all_score_lists,
    }


# ── Run TTA over full eval set ────────────────────────────────────────────────

def run_tta_evaluation(dataset_name="pepper_eval", score_thresh=0.3):
    cfg = build_cfg(score_thresh=score_thresh)
    predictor = DefaultPredictor(cfg)
    augmentations = get_augmentations()

    dataset = DatasetCatalog.get(dataset_name)
    print(f"\nRunning TTA on {len(dataset)} images from '{dataset_name}'")
    print(f"Augmentations: {[a[0] for a in augmentations]}\n")

    results = []
    uncertain_images = []

    for sample in tqdm(dataset, desc="TTA inference"):
        image_path = sample["file_name"]
        image_id   = sample["image_id"]

        result = tta_single_image(image_path, predictor, augmentations)

        entry = {
            "image_id":        image_id,
            "file_name":       image_path,
            "mean_score":      float(result["mean_score"]),
            "std_score":       float(result["std_score"]),    # uncertainty
            "mean_detections": float(result["mean_detections"]),
            "std_detections":  float(result["std_detections"]),
            "n_detections_original": len(
                result["orig_output"]["instances"].scores
            ),
        }
        results.append(entry)

        # Flag images where model is uncertain (high std)
        if result["std_score"] > 0.05:
            uncertain_images.append(entry)

    # ── Summary statistics ────────────────────────────────────────────────────
    std_scores = np.array([r["std_score"] for r in results])
    mean_scores = np.array([r["mean_score"] for r in results])

    print("\n" + "="*55)
    print("TTA UNCERTAINTY SUMMARY")
    print("="*55)
    print(f"Dataset:              {dataset_name}")
    print(f"Images evaluated:     {len(results)}")
    print(f"Augmentations used:   {len(augmentations)}")
    print(f"\nConfidence scores (mean across augmentations):")
    print(f"  Mean:  {mean_scores.mean():.4f}")
    print(f"  Std:   {mean_scores.std():.4f}")
    print(f"\nUncertainty (std of scores across augmentations):")
    print(f"  Mean:  {std_scores.mean():.4f}")
    print(f"  Max:   {std_scores.max():.4f}")
    print(f"  Min:   {std_scores.min():.4f}")
    print(f"\nHigh-uncertainty images (std > 0.05): {len(uncertain_images)}")
    print("="*55)

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = os.path.join(cfg.OUTPUT_DIR, "tta_uncertainty_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results, std_scores


# ── Standard COCO AP evaluation (original image only) ────────────────────────

def run_coco_evaluation(dataset_name="pepper_eval"):
    """
    Run standard COCO evaluation using original images only.
    This gives you the AP numbers to compare against baseline.
    """
    cfg = build_cfg(score_thresh=0.0)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    from detectron2.engine import DefaultTrainer
    from detectron2.checkpoint import DetectionCheckpointer

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    loader    = build_detection_test_loader(cfg, dataset_name)

    print(f"\nRunning COCO evaluation on '{dataset_name}'...")
    results = inference_on_dataset(model, loader, evaluator)
    print(results)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TTA Uncertainty Quantification")
    parser.add_argument("--mode", choices=["tta", "coco", "both"], default="both",
                        help="tta = uncertainty only, coco = AP metrics, both = run both")
    parser.add_argument("--dataset", default="pepper_eval",
                        help="Dataset split to evaluate on")
    parser.add_argument("--thresh", type=float, default=0.3,
                        help="Score threshold for TTA predictions")
    args = parser.parse_args()

    if args.mode in ("tta", "both"):
        results, uncertainties = run_tta_evaluation(
            dataset_name=args.dataset,
            score_thresh=args.thresh
        )

    if args.mode in ("coco", "both"):
        coco_results = run_coco_evaluation(dataset_name=args.dataset)