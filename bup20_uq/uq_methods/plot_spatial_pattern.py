import matplotlib.pyplot as plt
import json
import numpy as np

with open("output/instance_calibration_data.json") as f:
    data = json.load(f)

xs = np.array([d["norm_x"] for d in data])
ys = np.array([d["norm_y"] for d in data])
uncertainty = np.array([1 - d["score"] for d in data])

fig, ax = plt.subplots(figsize=(7,7))
sc = ax.scatter(xs, ys, c=uncertainty, cmap='hot', s=40, edgecolor='black', alpha=0.7)
ax.invert_yaxis()  # image coords: top=0
plt.colorbar(sc, label="Uncertainty (1 - confidence)")
ax.set_xlabel("Normalized X position (0=left, 1=right)")
ax.set_ylabel("Normalized Y position (0=top, 1=bottom)")
ax.set_title("Spatial Distribution of Instance Uncertainty")
plt.tight_layout()
plt.savefig("output/spatial_uncertainty_pattern.png", dpi=150)