import matplotlib.pyplot as plt
import json
import numpy as np

with open("output/instance_calibration_data.json") as f:
    data = json.load(f)

classes = sorted(set(d["class_name"] for d in data))
fig, ax = plt.subplots(figsize=(8,5))

class_uncertainty = {}
for c in classes:
    scores = [1 - d["score"] for d in data if d["class_name"] == c]  # uncertainty proxy
    class_uncertainty[c] = np.mean(scores)

# Rank by uncertainty
ranked = sorted(class_uncertainty.items(), key=lambda x: -x[1])
names = [r[0] for r in ranked]
vals  = [r[1] for r in ranked]

ax.barh(names, vals, color='#0FA3B1')
ax.set_xlabel("Mean Uncertainty (1 - confidence)")
ax.set_title("Class Ranking by Uncertainty")
plt.tight_layout()
plt.savefig("output/uncertainty_by_class.png", dpi=150)

print("Ranking (most to least uncertain):")
for name, val in ranked:
    print(f"  {name}: {val:.4f}")