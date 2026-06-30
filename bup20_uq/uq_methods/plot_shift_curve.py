import matplotlib.pyplot as plt
import json

with open("output/distribution_shift_results.json") as f:
    results = json.load(f)

angles = sorted([int(k) for k in results.keys()])
confs = [results[str(a)]["mean_confidence"] for a in angles]
dets  = [results[str(a)]["mean_detections"] for a in angles]

fig, ax1 = plt.subplots(figsize=(7,5))
ax1.plot(angles, confs, marker='o', color='#2E86AB', label='Mean Confidence')
ax1.set_xlabel("Rotation angle (degrees)")
ax1.set_ylabel("Mean confidence", color='#2E86AB')
ax2 = ax1.twinx()
ax2.plot(angles, dets, marker='s', color='#EF4444', label='Mean detections')
ax2.set_ylabel("Mean detections per image", color='#EF4444')
plt.title("Model Behaviour Under Distribution Shift (Rotation)")
fig.tight_layout()
plt.savefig("output/shift_robustness_curve.png", dpi=150)