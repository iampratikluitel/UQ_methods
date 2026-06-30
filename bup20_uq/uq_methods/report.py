"""Build a single comparison report across UQ methods."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from calibration_metrics import compute_ace, compute_ece
from plotting import (
    plot_method_uncertainty_comparison,
    plot_side_by_side_reliability,
)


def _load_method_uncertainty(method: str, report_dir: Path) -> float | None:
    """Return a scalar mean uncertainty for cross-method comparison."""
    if method == "baseline":
        data_path = report_dir / "baseline" / "instance_calibration_data.json"
        if not data_path.exists():
            data_path = report_dir.parent / "instance_calibration_data.json"
        if not data_path.exists():
            return None
        with open(data_path) as f:
            data = json.load(f)
        return sum(1 - d["score"] for d in data) / len(data)

    if method == "tta":
        path = report_dir.parent / "tta" / "tta_uncertainty_results.json"
    elif method == "mc_dropout":
        path = report_dir.parent / "mc_dropout" / "mc_dropout_uncertainty_results.json"
    else:
        return None

    if not path.exists():
        return None
    with open(path) as f:
        results = json.load(f)
    return sum(r["std_score"] for r in results) / len(results)


def generate_comparison_report(
    report_dir: Path,
    methods_run: list[str],
    calibration_metrics: dict[str, dict],
    shift_results: dict | None = None,
) -> Path:
    """Write markdown summary + side-by-side comparison plots."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # Side-by-side reliability (methods with instance-level calibration data)
    reliability_data = {}
    for method, metrics in calibration_metrics.items():
        data_path = report_dir / method / "instance_calibration_data.json"
        if not data_path.exists() and method == "baseline":
            data_path = report_dir.parent / "instance_calibration_data.json"
        if not data_path.exists():
            continue
        with open(data_path) as f:
            data = json.load(f)
        import numpy as np
        conf = np.array([d["score"] for d in data])
        acc = np.array([1.0 if d["correct"] else 0.0 for d in data])
        ece, bins = compute_ece(conf, acc)
        reliability_data[method] = (ece, bins)

    if reliability_data:
        plot_side_by_side_reliability(
            reliability_data,
            save_path=str(report_dir / "comparison_reliability.png"),
        )

    # Method uncertainty bar chart
    method_uncertainty = {}
    for method in methods_run:
        val = _load_method_uncertainty(method, report_dir)
        if val is not None:
            method_uncertainty[method] = val

    if len(method_uncertainty) >= 2:
        plot_method_uncertainty_comparison(
            method_uncertainty,
            save_path=str(report_dir / "comparison_method_uncertainty.png"),
        )

    # Markdown report
    lines = [
        "# UQ Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Methods run",
        "",
    ]
    for m in methods_run:
        lines.append(f"- {m}")

    lines += ["", "## Calibration metrics (instance-level)", "", "| Method | ECE | ACE | Accuracy | Mean conf | Conf gap |", "|---|---:|---:|---:|---:|---:|"]

    for method, m in calibration_metrics.items():
        lines.append(
            f"| {method} | {m['ece']:.4f} | {m['ace']:.4f} | {m['accuracy']:.4f} | "
            f"{m['mean_confidence']:.4f} | {m['confidence_gap']:+.4f} |"
        )

    if method_uncertainty:
        lines += ["", "## Image-level uncertainty (cross-method)", ""]
        for method, val in sorted(method_uncertainty.items(), key=lambda x: -x[1]):
            lines.append(f"- **{method}**: {val:.4f}")

    for method, m in calibration_metrics.items():
        ranking = m.get("class_uncertainty_ranking", [])
        if ranking:
            lines += [f"", f"### {method} — class uncertainty ranking", ""]
            for name, val in ranking:
                lines.append(f"1. {name}: {val:.4f}")

    if shift_results:
        lines += ["", "## Distribution shift (rotation)", "", "| Angle | Mean conf | Mean dets |", "|---:|---:|---:|"]
        for angle in sorted(shift_results.keys(), key=int):
            r = shift_results[angle]
            lines.append(f"| {angle}° | {r['mean_confidence']:.4f} | {r['mean_detections']:.2f} |")

    lines += [
        "",
        "## Figures",
        "",
        "- `comparison_reliability.png` — side-by-side reliability diagrams",
        "- `comparison_method_uncertainty.png` — mean uncertainty per method",
        "- `<method>/reliability_diagram.png` — per-method calibration",
        "- `<method>/calibration_matrix.png` — per-class ECE",
        "- `<method>/uncertainty_by_class.png` — class ranking",
        "- `<method>/spatial_uncertainty.png` — spatial uncertainty map",
        "- `shift_robustness_curve.png` — rotation shift curve",
        "",
        "## Notes",
        "",
        "- **baseline** uses detector confidence (1 − score) as uncertainty proxy.",
        "- **tta** / **mc_dropout** currently report image-level `std_score` across augmentations/passes.",
        "- Per-instance UQ from TTA/MC Dropout and **ensemble** are planned next.",
        "- OOD detection (AUROC) and AP/mIoU under shift require additional eval hooks.",
    ]

    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path
