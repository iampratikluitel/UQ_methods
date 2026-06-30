"""Build a single comparison report across UQ methods."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from calibration_metrics import compute_ace, compute_ece
from plotting import (
    plot_method_uncertainty_comparison,
    plot_side_by_side_reliability,
)


def _load_instance_data(method: str, report_dir: Path) -> list | None:
    data_path = report_dir / method / "instance_calibration_data.json"
    if not data_path.exists() and method == "baseline":
        data_path = report_dir.parent / "instance_calibration_data.json"
    if not data_path.exists():
        return None
    with open(data_path) as f:
        return json.load(f)


def _load_method_uncertainty(method: str, report_dir: Path) -> float | None:
    """Mean per-instance uncertainty (UQ std for TTA/MC, 1−score for baseline)."""
    data = _load_instance_data(method, report_dir)
    if not data:
        return None
    if "uncertainty" in data[0]:
        return sum(d["uncertainty"] for d in data) / len(data)
    return sum(1 - d["score"] for d in data) / len(data)


def generate_comparison_report(
    report_dir: Path,
    methods_run: list[str],
    calibration_metrics: dict[str, dict],
    shift_results: dict | None = None,
) -> Path:
    """Write markdown summary + side-by-side comparison plots."""
    report_dir.mkdir(parents=True, exist_ok=True)

    reliability_data = {}
    for method in methods_run:
        if method not in calibration_metrics:
            continue
        data = _load_instance_data(method, report_dir)
        if not data:
            continue
        conf = np.array([d["score"] for d in data])
        acc = np.array([1.0 if d["correct"] else 0.0 for d in data])
        ece, bins = compute_ece(conf, acc)
        reliability_data[method] = (ece, bins)

    if reliability_data:
        plot_side_by_side_reliability(
            reliability_data,
            save_path=str(report_dir / "comparison_reliability.png"),
        )

    method_uncertainty = {}
    for method in methods_run:
        val = _load_method_uncertainty(method, report_dir)
        if val is not None:
            method_uncertainty[method] = val

    if len(method_uncertainty) >= 2:
        plot_method_uncertainty_comparison(
            method_uncertainty,
            save_path=str(report_dir / "comparison_method_uncertainty.png"),
            title="Mean per-instance uncertainty by method",
        )

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

    lines += [
        "",
        "## Calibration metrics (instance-level)",
        "",
        "| Method | ECE | ACE | Accuracy | Mean conf | Conf gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in methods_run:
        if method not in calibration_metrics:
            continue
        m = calibration_metrics[method]
        lines.append(
            f"| {method} | {m['ece']:.4f} | {m['ace']:.4f} | {m['accuracy']:.4f} | "
            f"{m['mean_confidence']:.4f} | {m['confidence_gap']:+.4f} |"
        )

    if method_uncertainty:
        lines += ["", "## Mean per-instance uncertainty", ""]
        for method, val in sorted(method_uncertainty.items(), key=lambda x: -x[1]):
            note = "(1 − confidence)" if method == "baseline" else "(std across augs/passes)"
            lines.append(f"- **{method}**: {val:.4f} {note}")

    for method in methods_run:
        if method not in calibration_metrics:
            continue
        ranking = calibration_metrics[method].get("class_uncertainty_ranking", [])
        if ranking:
            lines += ["", f"### {method} — class uncertainty ranking", ""]
            for name, val in ranking:
                lines.append(f"- {name}: {val:.4f}")

    if shift_results:
        lines += [
            "", "## Distribution shift (rotation)", "",
            "| Angle | Mean conf | Mean dets |", "|---:|---:|---:|",
        ]
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
        "- All methods use the same pipeline: per-instance GT matching → ECE/ACE/confidence gap.",
        "- **baseline** confidence = raw detector score; uncertainty proxy = 1 − score.",
        "- **tta** confidence = mean score across spatial-preserving augmentations per instance.",
        "- **mc_dropout** confidence = mean score across T stochastic passes per instance.",
        "- TTA instance calibration uses: original, hflip, vflip, brightness± (no rotation).",
    ]

    report_path = report_dir / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return report_path
