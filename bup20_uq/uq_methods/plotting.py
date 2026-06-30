"""Shared plotting helpers for UQ benchmark reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_class_uncertainty_ranking(instances, save_path, uncertainty_key="score",
                                   invert=True, title="Class Ranking by Uncertainty"):
    """
    Rank classes by mean uncertainty.
    For baseline confidence, use invert=True (uncertainty = 1 - score).
    For MC/TTA, pass invert=False and uncertainty_key='std_score'.
    """
    classes = sorted({d["class_name"] for d in instances})
    class_uncertainty = {}
    for cls in classes:
        if uncertainty_key == "score" and invert:
            vals = [1 - d["score"] for d in instances if d["class_name"] == cls]
        else:
            vals = [d[uncertainty_key] for d in instances if d["class_name"] == cls]
        if vals:
            class_uncertainty[cls] = float(np.mean(vals))

    ranked = sorted(class_uncertainty.items(), key=lambda x: -x[1])
    names = [r[0] for r in ranked]
    vals = [r[1] for r in ranked]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color="#0FA3B1", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Mean uncertainty")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return ranked


def plot_spatial_uncertainty(instances, save_path, uncertainty_key="score", invert=True,
                             title="Spatial Distribution of Instance Uncertainty"):
    xs = np.array([d["norm_x"] for d in instances])
    ys = np.array([d["norm_y"] for d in instances])
    if uncertainty_key == "score" and invert:
        uncertainty = np.array([1 - d["score"] for d in instances])
    else:
        uncertainty = np.array([d[uncertainty_key] for d in instances])

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(xs, ys, c=uncertainty, cmap="hot", s=40, edgecolor="black", alpha=0.7)
    ax.invert_yaxis()
    plt.colorbar(sc, label="Uncertainty")
    ax.set_xlabel("Normalized X (0=left, 1=right)")
    ax.set_ylabel("Normalized Y (0=top, 1=bottom)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_shift_robustness(shift_results, save_path, title="Model Behaviour Under Distribution Shift"):
    """Plot confidence and detection count vs rotation angle."""
    angles = sorted(shift_results.keys(), key=lambda x: int(x))
    confs = [shift_results[str(a)]["mean_confidence"] for a in angles]
    dets = [shift_results[str(a)]["mean_detections"] for a in angles]
    angle_labels = [int(a) for a in angles]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(angle_labels, confs, marker="o", color="#2E86AB", label="Mean confidence")
    ax1.set_xlabel("Rotation angle (degrees)")
    ax1.set_ylabel("Mean confidence", color="#2E86AB")
    ax2 = ax1.twinx()
    ax2.plot(angle_labels, dets, marker="s", color="#EF4444", label="Mean detections")
    ax2.set_ylabel("Mean detections per image", color="#EF4444")
    plt.title(title)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_method_uncertainty_comparison(method_stats, save_path,
                                       title="UQ Method Comparison (mean image-level uncertainty)"):
    """
    method_stats: {"baseline": 0.12, "tta": 0.08, ...}
    """
    names = list(method_stats.keys())
    vals = [method_stats[n] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, vals, color=["#2E86AB", "#0FA3B1", "#EF4444", "#F4A261"][: len(names)],
                  edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean uncertainty")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_side_by_side_reliability(bin_data_by_method, save_path, n_bins_label="ECE"):
    """
    bin_data_by_method: {method_name: (ece_value, bin_data)}
    """
    methods = list(bin_data_by_method.keys())
    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, method in zip(axes[0], methods):
        ece, bin_data = bin_data_by_method[method]
        confs = [b["conf"] for b in bin_data if b["acc"] is not None]
        accs = [b["acc"] for b in bin_data if b["acc"] is not None]
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        ax.bar(confs, accs, width=0.08, alpha=0.7, edgecolor="black", color="#2E86AB")
        for c, a in zip(confs, accs):
            ax.plot([c, c], [c, a], color="red", alpha=0.5, linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{method}\n({n_bins_label}={ece:.3f})")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
