# UQ Benchmark Checklist

Supervisor deliverables mapped to code status. Run everything via:

```bash
cd bup20_uq
python uq_methods/run_uq_benchmark.py --methods baseline tta mc_dropout
```

Quick report from cached baseline data (no GPU inference):

```bash
python uq_methods/run_uq_benchmark.py --quick
```

---

## Phase 0 — Foundation

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| Per-detection confidence + GT matching (IoU) | ✅ Done | `extract_instance_confidence.py` | Greedy IoU≥0.5 matching |
| Shared paths / config | ✅ Done | `config.py` | Set `BUP20_IMAGE_ROOT` locally if paths differ |
| Master benchmark runner | ✅ Done | `run_uq_benchmark.py` | Runs all methods + report |
| Comparison report (markdown + plots) | ✅ Done | `report.py` | `output/uq_report/REPORT.md` |

---

## Phase 1 — Calibration

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| ECE (equal-width bins) | ✅ Done | `calibration_metrics.py` | |
| ACE (adaptive bins) | ✅ Done | `calibration_metrics.py` | |
| Reliability diagram | ✅ Done | `calibration_metrics.py` | Side-by-side in report |
| Confidence histogram | ✅ Done | `calibration_metrics.py` | |
| Per-class ECE (calibration matrix) | ✅ Done | `calibration_metrics.py` | `calibration_matrix.png` |
| Quantitative ECE/ACE in report | ✅ Done | `run_calibration_analysis.py` | |

**Gap:** Image-level `tta.py` / `mc_dropout.py` scripts remain for quick summaries; calibration uses `extract_instance_uq.py`.

---

## Phase 2 — Uncertainty measures

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| Baseline uncertainty (1 − confidence) | ✅ Done | `extract_instance_confidence.py` | Per instance |
| TTA instance calibration (per-detection) | ✅ Done | `extract_instance_uq.py` | Spatial-preserving augs only |
| MC Dropout instance calibration | ✅ Done | `extract_instance_uq.py` | Cluster across T passes by IoU |
| ECE/ACE/confidence gap for TTA & MC | ✅ Done | `run_uq_benchmark.py` | Same pipeline as baseline |
| Ensemble uncertainty | ❌ Todo | — | Stub in master script |
| Class uncertainty ranking | ✅ Done | `plotting.py` | Histogram / bar chart |
| Spatial uncertainty map | ✅ Done | `plotting.py` | norm_x / norm_y scatter |
| Occlusion vs certainty pattern | ❌ Todo | — | Need mask-overlap proxy |

---

## Phase 3 — Distribution shift & robustness

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| Rotation shift eval | ⚠️ Partial | `distribution_shift_evaluation.py` | Confidence + det count only |
| Shift robustness curve | ✅ Done | `plotting.py` | |
| AP under shift | ❌ Todo | — | Need COCOEvaluator on rotated images |
| mIoU under shift | ❌ Todo | — | Need mask rasterization |
| Brightness / blur / noise shifts | ❌ Todo | — | Augmentations exist in `tta.py` |
| "Is method robust?" summary | ⚠️ Partial | `report.py` | Needs AP + uncertainty correlation |

---

## Phase 4 — OOD detection

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| ID vs OOD split definition | ❌ Todo | — | e.g. clean eval vs heavy corruption |
| AUROC / AUPR for OOD | ❌ Todo | — | Use uncertainty as detector score |
| OOD examples in report | ❌ Todo | — | |

---

## Phase 5 — Qualitative

| Task | Status | Module | Notes |
|------|--------|--------|-------|
| GT vs prediction panels | ✅ Done | `ground_truth.py` | Wired in master script |
| Uncertainty overlay on predictions | ❌ Todo | — | Color instances by uncertainty |
| Best / worst case gallery | ❌ Todo | — | Sort by uncertainty |

---

## Known issues / fixes needed

1. **Hardcoded HPC image paths** in `data/annotations_*.json` — set `export BUP20_IMAGE_ROOT=/your/local/path` before running.
2. **`register_dataset.py`** still uses cluster paths — update when running locally.
3. **TTA / MC Dropout** uncertainty is per-image, not per-detection — limits fair comparison with baseline calibration.
4. **`plot_class_uncertainty.py`**, **`plot_spatial_pattern.py`**, **`plot_shift_curve.py`** are legacy standalone scripts — use `run_uq_benchmark.py` instead.
5. **Ensemble** not implemented — can use `model_*.pth` checkpoints when ready.

---

## Recommended next steps (priority order)

1. Run `python uq_methods/run_uq_benchmark.py --quick` to validate report pipeline.
2. On cluster: `python uq_methods/run_uq_benchmark.py --methods baseline tta mc_dropout`.
3. Upgrade TTA/MC to **per-instance** uncertainty (cluster detections across passes by IoU). ✅ Done — see `extract_instance_uq.py`
4. Add **AP + mIoU** under rotation in `distribution_shift_evaluation.py`.
5. Add **OOD AUROC** using high-shift images as OOD.
6. Implement **ensemble** from saved checkpoints.

---

## Output layout

```
output/
├── instance_calibration_data.json      # baseline per-instance records
├── uq_report/
│   ├── REPORT.md                       # main comparison report
│   ├── comparison_reliability.png
│   ├── comparison_method_uncertainty.png
│   ├── shift_robustness_curve.png
│   ├── baseline/
│   │   ├── reliability_diagram.png
│   │   ├── calibration_matrix.png
│   │   ├── uncertainty_by_class.png
│   │   └── spatial_uncertainty.png
│   └── qualitative/
│       └── gt_vs_pred_*.png
├── tta/tta_uncertainty_results.json
└── mc_dropout/mc_dropout_uncertainty_results.json
```
