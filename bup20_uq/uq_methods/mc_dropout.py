
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import register_dataset  # noqa: F401
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model


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
    cfg.OUTPUT_DIR = "/home/s29pluit/projects/bup20_uq/output/mc_dropout"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


# ── Dropout injection ─────────────────────────────────────────────────────────

def add_dropout_to_model(model, dropout_rate=0.5):
    """
    Injects Dropout layers into the ROI box head and mask head.
    These are the prediction heads — injecting here gives uncertainty
    in both classification scores and mask predictions.
    """
    # Inject into box head FC layers
    box_head = model.roi_heads.box_head
    layers = list(box_head.fc_relu1.children()) if hasattr(box_head, 'fc_relu1') else []

    # Walk all linear layers in box predictor and add dropout after each
    for name, module in model.roi_heads.named_modules():
        if isinstance(module, nn.Linear):
            # We patch the parent module to insert dropout
            pass  # handled below via forward hook

    return model


def enable_dropout(model):
    """
    Sets all Dropout layers to train mode so they remain active
    during inference. This is the core of MC Dropout.
    Without this, model.eval() disables all dropout.
    """
    dropout_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()
            dropout_count += 1

    if dropout_count == 0:
        print("WARNING: No Dropout layers found in model.")
        print("Mask R-CNN R-50 does not have dropout by default.")
        print("Adding dropout to ROI heads now...")
        model = inject_dropout_roi_heads(model, dropout_rate=0.5)

    return model, dropout_count


def inject_dropout_roi_heads(model, dropout_rate=0.3):
    """
    Mask R-CNN R-50 does not have dropout by default.
    This patches the box predictor and mask head to include dropout,
    then the model needs to be put in eval() with dropout kept in train().
    """
    # Patch box head — insert dropout after each FC layer
    original_box_forward = model.roi_heads.box_predictor.forward

    def box_forward_with_dropout(x):
        x = torch.nn.functional.dropout(x, p=dropout_rate, training=True)
        return original_box_forward(x)

    model.roi_heads.box_predictor.forward = box_forward_with_dropout

    # Patch mask head — insert dropout in the conv layers
    original_mask_forward = model.roi_heads.mask_head.forward

    def mask_forward_with_dropout(x, *args, **kwargs):
        x = torch.nn.functional.dropout2d(x, p=dropout_rate, training=True)
        return original_mask_forward(x, *args, **kwargs)

    model.roi_heads.mask_head.forward = mask_forward_with_dropout

    print(f"Dropout (rate={dropout_rate}) injected into box_predictor and mask_head.")
    return model


# ── Preprocessing (matches Detectron2 test pipeline) ─────────────────────────

def preprocess_image(image_path, cfg):
    """Load and preprocess image the same way Detectron2 does at test time."""
    from detectron2.data import transforms as T
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read: {image_path}")

    # Resize to match test input size
    aug = T.ResizeShortestEdge(
        short_edge_length=[800],
        max_size=1333,
        sample_style="choice"
    )
    image_rgb = image[:, :, ::-1]  # BGR → RGB
    input_image = aug.get_transform(image_rgb).apply_image(image_rgb)
    input_tensor = torch.as_tensor(
        input_image.astype("float32").transpose(2, 0, 1)
    )
    return input_tensor, image.shape[:2]


# ── MC Dropout inference ──────────────────────────────────────────────────────

def mc_dropout_single_image(image_path, model, cfg, T=30):
    """
    Run T stochastic forward passes on a single image.

    Returns:
        mean_scores     : mean confidence per pass
        std_scores      : std of confidence (epistemic uncertainty)
        mean_n_dets     : mean number of detections across passes
        std_n_dets      : std of detection count (structural uncertainty)
        all_scores      : raw score list from each pass
    """
    input_tensor, orig_size = preprocess_image(image_path, cfg)

    inputs = [{
        "image": input_tensor,
        "height": orig_size[0],
        "width":  orig_size[1],
    }]

    all_scores      = []
    all_n_dets      = []
    all_mean_scores = []

    with torch.no_grad():
        for t in range(T):
            outputs = model(inputs)
            instances = outputs[0]["instances"]
            scores = instances.scores.cpu().numpy()

            all_scores.append(scores)
            all_n_dets.append(len(scores))
            all_mean_scores.append(scores.mean() if len(scores) > 0 else 0.0)

    all_mean_scores = np.array(all_mean_scores)
    all_n_dets      = np.array(all_n_dets, dtype=float)

    return {
        "mean_score":      float(all_mean_scores.mean()),
        "std_score":       float(all_mean_scores.std()),   # ← epistemic uncertainty
        "mean_detections": float(all_n_dets.mean()),
        "std_detections":  float(all_n_dets.std()),
        "all_scores":      [s.tolist() for s in all_scores],
        "T":               T,
    }


# ── Run MC Dropout over full eval set ────────────────────────────────────────

def run_mc_dropout_evaluation(
    dataset_name="pepper_eval",
    T=30,
    dropout_rate=0.3,
    score_thresh=0.3
):
    cfg = build_cfg(score_thresh=score_thresh)

    # Build model and load trained weights
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Inject dropout and keep it active
    model = inject_dropout_roi_heads(model, dropout_rate=dropout_rate)
    model.eval()
    # NOTE: do NOT call enable_dropout() after inject — the patched
    # forward functions already have training=True hardcoded

    dataset = DatasetCatalog.get(dataset_name)
    print(f"\nRunning MC Dropout on {len(dataset)} images")
    print(f"Forward passes per image (T): {T}")
    print(f"Dropout rate: {dropout_rate}\n")

    results      = []
    uncertain_images = []

    for sample in tqdm(dataset, desc="MC Dropout inference"):
        image_path = sample["file_name"]
        image_id   = sample["image_id"]

        result = mc_dropout_single_image(image_path, model, cfg, T=T)

        entry = {
            "image_id":        image_id,
            "file_name":       image_path,
            "mean_score":      result["mean_score"],
            "std_score":       result["std_score"],       # uncertainty
            "mean_detections": result["mean_detections"],
            "std_detections":  result["std_detections"],
            "T":               T,
        }
        results.append(entry)

        if result["std_score"] > 0.05:
            uncertain_images.append(entry)

    # ── Summary ───────────────────────────────────────────────────────────────
    std_scores  = np.array([r["std_score"]  for r in results])
    mean_scores = np.array([r["mean_score"] for r in results])

    print("\n" + "="*55)
    print("MC DROPOUT UNCERTAINTY SUMMARY")
    print("="*55)
    print(f"Dataset:              {dataset_name}")
    print(f"Images evaluated:     {len(results)}")
    print(f"Forward passes (T):   {T}")
    print(f"Dropout rate:         {dropout_rate}")
    print(f"\nConfidence scores (mean across T passes):")
    print(f"  Mean:  {mean_scores.mean():.4f}")
    print(f"  Std:   {mean_scores.std():.4f}")
    print(f"\nUncertainty (std across T passes):")
    print(f"  Mean:  {std_scores.mean():.4f}")
    print(f"  Max:   {std_scores.max():.4f}")
    print(f"  Min:   {std_scores.min():.4f}")
    print(f"\nHigh-uncertainty images (std > 0.05): {len(uncertain_images)}")
    print("="*55)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(cfg.OUTPUT_DIR, "mc_dropout_uncertainty_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results, std_scores


# ── Standard COCO AP evaluation with dropout active ──────────────────────────

def run_coco_evaluation_with_dropout(
    dataset_name="pepper_eval",
    dropout_rate=0.3
):
    """
    Runs standard COCO AP evaluation with dropout active.
    Useful to check whether dropout hurts segmentation accuracy.
    """
    cfg = build_cfg(score_thresh=0.0)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model = inject_dropout_roi_heads(model, dropout_rate=dropout_rate)
    model.eval()

    evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
    loader    = build_detection_test_loader(cfg, dataset_name)

    print(f"\nRunning COCO evaluation with MC Dropout on '{dataset_name}'...")
    results = inference_on_dataset(model, loader, evaluator)
    print(results)
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MC Dropout Uncertainty Quantification")
    parser.add_argument("--mode",     choices=["mc", "coco", "both"], default="both")
    parser.add_argument("--dataset",  default="pepper_eval")
    parser.add_argument("--T",        type=int,   default=30,
                        help="Number of stochastic forward passes")
    parser.add_argument("--dropout",  type=float, default=0.3,
                        help="Dropout rate (0.1–0.5 recommended)")
    parser.add_argument("--thresh",   type=float, default=0.3,
                        help="Score threshold for predictions")
    args = parser.parse_args()

    if args.mode in ("mc", "both"):
        results, uncertainties = run_mc_dropout_evaluation(
            dataset_name=args.dataset,
            T=args.T,
            dropout_rate=args.dropout,
            score_thresh=args.thresh,
        )

    if args.mode in ("coco", "both"):
        coco_results = run_coco_evaluation_with_dropout(
            dataset_name=args.dataset,
            dropout_rate=args.dropout,
        )