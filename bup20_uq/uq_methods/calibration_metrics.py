"""
calibration_metrics.py
/home/s29pluit/projects/bup20_uq/uq_methods/calibration_metrics.py
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_ece(confidences, accuracies, n_bins=10):
    """
    Expected Calibration Error — equal-width bins.

    confidences: array of predicted confidence scores [0,1]
    accuracies:  array of 1/0 indicating if prediction was correct
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_data.append({"conf": (lo+hi)/2, "acc": None, "count": 0})
            continue

        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        bin_weight = mask.sum() / len(confidences)

        ece += bin_weight * abs(bin_acc - bin_conf)
        bin_data.append({"conf": bin_conf, "acc": bin_acc, "count": mask.sum()})

    return ece, bin_data


def compute_ace(confidences, accuracies, n_bins=10):
    """
    Adaptive Calibration Error — equal-count (quantile) bins.
    More reliable when confidence scores are not uniformly spread.
    """
    n = len(confidences)
    sorted_idx = np.argsort(confidences)
    conf_sorted = confidences[sorted_idx]
    acc_sorted  = accuracies[sorted_idx]

    bin_size = n // n_bins
    ace = 0.0
    bin_data = []

    for i in range(n_bins):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n_bins - 1 else n
        if end <= start:
            continue
        bin_conf = conf_sorted[start:end].mean()
        bin_acc  = acc_sorted[start:end].mean()
        ace += abs(bin_acc - bin_conf) / n_bins
        bin_data.append({"conf": bin_conf, "acc": bin_acc, "count": end - start})

    return ace, bin_data


def plot_reliability_diagram(bin_data, title="Reliability Diagram", save_path=None):
    """
    Plots confidence vs accuracy per bin with a perfect-calibration diagonal.
    """
    confs = [b["conf"] for b in bin_data if b["acc"] is not None]
    accs  = [b["acc"]  for b in bin_data if b["acc"] is not None]
    counts = [b["count"] for b in bin_data if b["acc"] is not None]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    # Bar plot showing actual accuracy per confidence bin
    ax.bar(confs, accs, width=0.08, alpha=0.7, edgecolor="black",
           color="#2E86AB", label="Model")

    # Gap visualization (miscalibration)
    for c, a in zip(confs, accs):
        ax.plot([c, c], [c, a], color="red", alpha=0.5, linewidth=1)

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_histogram(confidences, title="Confidence Distribution", save_path=None):
    """Histogram of how confident the model is across all predictions."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=20, color="#0FA3B1", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Number of predictions")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()