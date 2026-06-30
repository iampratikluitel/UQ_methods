"""
run_calibration_analysis.py
"""
import json
import numpy as np
from calibration_metrics import compute_ece, compute_ace, plot_reliability_diagram, plot_confidence_histogram

with open("/home/s29pluit/projects/bup20_uq/output/instance_calibration_data.json") as f:
    data = json.load(f)

confidences = np.array([d["score"] for d in data])
accuracies  = np.array([1.0 if d["correct"] else 0.0 for d in data])

ece, ece_bins = compute_ece(confidences, accuracies, n_bins=10)
ace, ace_bins = compute_ace(confidences, accuracies, n_bins=10)

print(f"ECE:  {ece:.4f}")
print(f"ACE:  {ace:.4f}")
print(f"Mean confidence: {confidences.mean():.4f}")
print(f"Overall accuracy: {accuracies.mean():.4f}")
print(f"Gap (overconfidence if positive): {confidences.mean() - accuracies.mean():.4f}")

plot_reliability_diagram(ece_bins, title=f"Reliability Diagram (ECE={ece:.3f})",
                          save_path="/home/s29pluit/projects/bup20_uq/output/reliability_diagram.png")
plot_confidence_histogram(confidences,
                           save_path="/home/s29pluit/projects/bup20_uq/output/confidence_histogram.png")


