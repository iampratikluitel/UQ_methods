"""Shared paths and defaults for the UQ benchmark pipeline."""

from __future__ import annotations

import os
from pathlib import Path

# Project root: .../bup20_uq
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("BUP20_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("BUP20_OUTPUT_DIR", PROJECT_ROOT / "output"))
REPORT_DIR = OUTPUT_DIR / "uq_report"
METHOD_COMPARISON_DIR = OUTPUT_DIR / "method_comparison"

DEFAULT_WEIGHTS = Path(
    os.environ.get("BUP20_WEIGHTS", OUTPUT_DIR / "model_final.pth")
)
DEFAULT_DATASET = os.environ.get("BUP20_DATASET", "pepper_eval")
DEFAULT_SCORE_THRESH = float(os.environ.get("BUP20_SCORE_THRESH", "0.3"))

# HPC image root override (annotations JSON may store absolute cluster paths).
IMAGE_ROOT = os.environ.get("BUP20_IMAGE_ROOT", "")


def resolve_image_path(file_name: str) -> str:
    """Remap cluster image paths when running locally."""
    if not IMAGE_ROOT:
        return file_name
    marker = "CKA_sweet_pepper_2020_summer/"
    if marker in file_name:
        rel = file_name.split(marker, 1)[-1]
        return str(Path(IMAGE_ROOT) / rel)
    return file_name


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METHOD_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
