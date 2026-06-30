#!/usr/bin/env python3
"""
Master UQ benchmark runner.

Runs selected methods (baseline, tta, mc_dropout, ensemble) and produces
a single comparison report with plots side by side.

Usage (from bup20_uq/):
    python uq_methods/run_uq_benchmark.py --methods baseline tta mc_dropout
    python uq_methods/run_uq_benchmark.py --methods baseline --skip-shift
    python uq_methods/run_uq_benchmark.py --quick   # baseline only, reuse cached data
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Allow imports when run as script from project root
UQ_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UQ_DIR))
sys.path.insert(0, str(UQ_DIR.parent))

import register_dataset  # noqa: F401

from config import (
    DEFAULT_DATASET,
    DEFAULT_SCORE_THRESH,
    DEFAULT_WEIGHTS,
    OUTPUT_DIR,
    REPORT_DIR,
    ensure_dirs,
)
from distribution_shift_evaluation import run_distribution_shift
from extract_instance_confidence import extract_all_instance_data
from report import generate_comparison_report
from run_calibration_analysis import run_calibration_analysis


ALL_METHODS = ("baseline", "tta", "mc_dropout", "ensemble")


def run_baseline(dataset: str, weights: Path, report_dir: Path, score_thresh: float,
                 reuse_data: bool) -> dict:
    data_path = report_dir / "baseline" / "instance_calibration_data.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_data and (OUTPUT_DIR / "instance_calibration_data.json").exists():
        shutil.copy(OUTPUT_DIR / "instance_calibration_data.json", data_path)
        print(f"Reusing cached instance data → {data_path}")
    else:
        extract_all_instance_data(
            dataset_name=dataset,
            weights_path=str(weights),
            score_thresh=score_thresh,
            output_path=str(data_path),
        )

    return run_calibration_analysis(data_path, report_dir, method_name="baseline")


def run_tta(dataset: str, score_thresh: float) -> list:
    from tta import run_tta_evaluation
    results, _ = run_tta_evaluation(dataset_name=dataset, score_thresh=score_thresh)
    return results


def run_mc_dropout(dataset: str, T: int, dropout: float, score_thresh: float) -> list:
    from mc_dropout import run_mc_dropout_evaluation
    results, _ = run_mc_dropout_evaluation(
        dataset_name=dataset,
        T=T,
        dropout_rate=dropout,
        score_thresh=score_thresh,
    )
    return results


def run_ensemble(_dataset: str) -> None:
    raise NotImplementedError(
        "Ensemble UQ is not implemented yet. "
        "Train multiple checkpoints or use model_*.pth weights and add ensemble voting."
    )


def save_qualitative_samples(dataset: str, weights: Path, report_dir: Path, n: int = 5):
    """Save GT vs prediction panels for a few eval images."""
    import random
    import cv2
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from ground_truth import visualize_gt_vs_pred
    from config import resolve_image_path

    qual_dir = report_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = str(weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DEFAULT_SCORE_THRESH
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get(dataset)
    metadata = MetadataCatalog.get(dataset)
    samples = random.sample(dataset_dicts, min(n, len(dataset_dicts)))

    for sample in samples:
        sample = dict(sample)
        sample["file_name"] = resolve_image_path(sample["file_name"])
        out = qual_dir / f"gt_vs_pred_{sample['image_id']}.png"
        visualize_gt_vs_pred(sample, predictor, metadata, str(out))
        print(f"Saved qualitative: {out}")


def main():
    parser = argparse.ArgumentParser(description="Run UQ benchmark across methods")
    parser.add_argument("--methods", nargs="+", default=["baseline"],
                        help=f"UQ methods: {', '.join(ALL_METHODS)}")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--thresh", type=float, default=DEFAULT_SCORE_THRESH)
    parser.add_argument("--out", default=str(REPORT_DIR))
    parser.add_argument("--T", type=int, default=30, help="MC Dropout forward passes")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--skip-shift", action="store_true")
    parser.add_argument("--skip-qualitative", action="store_true")
    parser.add_argument("--reuse-data", action="store_true",
                        help="Reuse existing instance_calibration_data.json for baseline")
    parser.add_argument("--quick", action="store_true",
                        help="baseline + report only, reuse cached data, skip shift")
    args = parser.parse_args()

    if args.quick:
        args.methods = ["baseline"]
        args.reuse_data = True
        args.skip_shift = True

    ensure_dirs()
    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights)

    methods = args.methods
    for m in methods:
        if m not in ALL_METHODS:
            parser.error(f"Unknown method '{m}'. Choose from: {ALL_METHODS}")

    print("=" * 60)
    print("UQ BENCHMARK")
    print("=" * 60)
    print(f"Methods:  {methods}")
    print(f"Dataset:  {args.dataset}")
    print(f"Weights:  {weights}")
    print(f"Report:   {report_dir}")
    print("=" * 60)

    calibration_metrics: dict[str, dict] = {}

    for method in methods:
        print(f"\n>>> Running: {method}")
        try:
            if method == "baseline":
                calibration_metrics["baseline"] = run_baseline(
                    args.dataset, weights, report_dir, args.thresh, args.reuse_data
                )
            elif method == "tta":
                run_tta(args.dataset, args.thresh)
                # TTA is image-level; copy summary into report metrics placeholder
                tta_path = OUTPUT_DIR / "tta" / "tta_uncertainty_results.json"
                if tta_path.exists():
                    with open(tta_path) as f:
                        tta_data = json.load(f)
                    mean_unc = sum(d["std_score"] for d in tta_data) / len(tta_data)
                    calibration_metrics["tta"] = {
                        "method": "tta",
                        "n_instances": len(tta_data),
                        "ece": None,
                        "ace": None,
                        "mean_confidence": sum(d["mean_score"] for d in tta_data) / len(tta_data),
                        "accuracy": None,
                        "confidence_gap": None,
                        "note": f"image-level std_score mean={mean_unc:.4f}",
                    }
            elif method == "mc_dropout":
                run_mc_dropout(args.dataset, args.T, args.dropout, args.thresh)
                mc_path = OUTPUT_DIR / "mc_dropout" / "mc_dropout_uncertainty_results.json"
                if mc_path.exists():
                    with open(mc_path) as f:
                        mc_data = json.load(f)
                    mean_unc = sum(d["std_score"] for d in mc_data) / len(mc_data)
                    calibration_metrics["mc_dropout"] = {
                        "method": "mc_dropout",
                        "n_instances": len(mc_data),
                        "ece": None,
                        "ace": None,
                        "mean_confidence": sum(d["mean_score"] for d in mc_data) / len(mc_data),
                        "accuracy": None,
                        "confidence_gap": None,
                        "note": f"image-level std_score mean={mean_unc:.4f}",
                    }
            elif method == "ensemble":
                run_ensemble(args.dataset)
        except NotImplementedError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR in {method}: {e}")
            raise

    shift_results = None
    if not args.skip_shift:
        print("\n>>> Distribution shift evaluation")
        shift_results = run_distribution_shift(
            args.dataset, weights, report_dir
        )

    if not args.skip_qualitative and "baseline" in methods:
        print("\n>>> Qualitative GT vs prediction samples")
        try:
            save_qualitative_samples(args.dataset, weights, report_dir, n=5)
        except Exception as e:
            print(f"  Qualitative skipped: {e}")

    # Filter calibration_metrics for report table (only those with ECE)
    report_calibration = {
        k: v for k, v in calibration_metrics.items()
        if v.get("ece") is not None
    }
    generate_comparison_report(
        report_dir=report_dir,
        methods_run=[m for m in methods if m in calibration_metrics or m in ("tta", "mc_dropout")],
        calibration_metrics=report_calibration or calibration_metrics,
        shift_results=shift_results,
    )

    # Save full run config
    with open(report_dir / "run_config.json", "w") as f:
        json.dump({
            "methods": methods,
            "dataset": args.dataset,
            "weights": str(weights),
            "thresh": args.thresh,
            "calibration_metrics": calibration_metrics,
        }, f, indent=2)

    print("\nDone. Open REPORT.md in:", report_dir)


if __name__ == "__main__":
    main()
