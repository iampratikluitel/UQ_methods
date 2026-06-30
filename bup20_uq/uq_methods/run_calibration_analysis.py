"""
Run calibration analysis on per-instance confidence data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration_metrics import (
    compute_ace,
    compute_ece,
    compute_per_class_ece,
    plot_calibration_matrix,
    plot_confidence_histogram,
    plot_reliability_diagram,
)
from plotting import plot_class_uncertainty_ranking, plot_spatial_uncertainty


def run_calibration_analysis(
    data_path: Path,
    out_dir: Path,
    method_name: str = "baseline",
    n_bins: int = 10,
) -> dict:
    """Compute ECE/ACE and save calibration plots for one method."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    confidences = np.array([d["score"] for d in data])
    accuracies = np.array([1.0 if d["correct"] else 0.0 for d in data])

    ece, ece_bins = compute_ece(confidences, accuracies, n_bins=n_bins)
    ace, ace_bins = compute_ace(confidences, accuracies, n_bins=n_bins)
    class_names = sorted({d["class_name"] for d in data})
    per_class_ece = compute_per_class_ece(data, class_names, n_bins=n_bins)

    metrics = {
        "method": method_name,
        "n_instances": len(data),
        "ece": float(ece),
        "ace": float(ace),
        "mean_confidence": float(confidences.mean()),
        "accuracy": float(accuracies.mean()),
        "confidence_gap": float(confidences.mean() - accuracies.mean()),
        "per_class_ece": per_class_ece,
    }

    prefix = out_dir / method_name
    prefix.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(
        ece_bins,
        title=f"{method_name} — Reliability (ECE={ece:.3f})",
        save_path=str(prefix / "reliability_diagram.png"),
    )
    plot_confidence_histogram(
        confidences,
        title=f"{method_name} — Confidence Distribution",
        save_path=str(prefix / "confidence_histogram.png"),
    )
    plot_calibration_matrix(
        per_class_ece,
        title=f"{method_name} — Per-Class ECE",
        save_path=str(prefix / "calibration_matrix.png"),
    )
    ranked = plot_class_uncertainty_ranking(
        data,
        save_path=str(prefix / "uncertainty_by_class.png"),
        title=f"{method_name} — Class Uncertainty Ranking",
        uncertainty_key="uncertainty" if data and "uncertainty" in data[0] else "score",
        invert=method_name == "baseline",
    )
    plot_spatial_uncertainty(
        data,
        save_path=str(prefix / "spatial_uncertainty.png"),
        title=f"{method_name} — Spatial Uncertainty",
        uncertainty_key="uncertainty" if data and "uncertainty" in data[0] else "score",
        invert=method_name == "baseline",
    )

    metrics["class_uncertainty_ranking"] = ranked
    with open(prefix / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{method_name}] ECE={ece:.4f}  ACE={ace:.4f}  "
          f"acc={accuracies.mean():.4f}  conf={confidences.mean():.4f}")

    return metrics


if __name__ == "__main__":
    import argparse
    from config import OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(OUTPUT_DIR / "instance_calibration_data.json"))
    parser.add_argument("--out", default=str(OUTPUT_DIR / "uq_report"))
    parser.add_argument("--method", default="baseline")
    args = parser.parse_args()

    run_calibration_analysis(Path(args.data), Path(args.out), method_name=args.method)
