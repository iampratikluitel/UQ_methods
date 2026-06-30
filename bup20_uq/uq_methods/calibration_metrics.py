"""
calibration_metrics.py
/home/s29pluit/projects/bup20_uq/uq_methods/calibration_metrics.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def compute_per_class_ece(instances, class_names, n_bins=10):
    """Return {class_name: ece} calibration matrix entries."""
    import numpy as np

    matrix = {}
    for cls in class_names:
        subset = [d for d in instances if d["class_name"] == cls]
        if len(subset) < n_bins:
            continue
        conf = np.array([d["score"] for d in subset])
        acc = np.array([1.0 if d["correct"] else 0.0 for d in subset])
        ece, _ = compute_ece(conf, acc, n_bins=n_bins)
        matrix[cls] = ece
    return matrix


def plot_calibration_matrix(per_class_ece, title="Per-Class ECE (Calibration Matrix)", save_path=None):
    """Bar chart of ECE per class — higher = more miscalibrated."""
    if not per_class_ece:
        return

    classes = list(per_class_ece.keys())
    values = [per_class_ece[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, values, color="#2E86AB", edgecolor="black", alpha=0.85)
    ax.set_ylabel("ECE")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
