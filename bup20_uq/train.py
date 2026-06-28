import os
import register_dataset  # noqa: F401

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class Trainer(DefaultTrainer):
    """Custom trainer with COCO evaluator attached."""

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    # ── datasets ──────────────────────────────────────────────
    cfg.DATASETS.TRAIN = ("pepper_train",)
    cfg.DATASETS.TEST  = ("pepper_val",)

    # ── dataloader ────────────────────────────────────────────
    cfg.DATALOADER.NUM_WORKERS = 4

    # ── model ─────────────────────────────────────────────────
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # increased from 128

    # ── solver ────────────────────────────────────────────────
    cfg.SOLVER.IMS_PER_BATCH   = 2
    cfg.SOLVER.BASE_LR         = 0.00025
    cfg.SOLVER.MAX_ITER        = 5000       # was 200 — far too short
    cfg.SOLVER.STEPS           = (3000, 4000)  # LR decay points
    cfg.SOLVER.GAMMA           = 0.1           # LR multiplier at each step
    cfg.SOLVER.WARMUP_ITERS    = 500           # gradual LR warmup
    cfg.SOLVER.WARMUP_METHOD   = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = 500         # save every 500 iters

    # ── evaluation ────────────────────────────────────────────
    cfg.TEST.EVAL_PERIOD = 500  # evaluate on val every 500 iters

    # ── output ────────────────────────────────────────────────
    cfg.OUTPUT_DIR = "./output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def main():
    cfg = setup_cfg()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)  # resume=False = fresh start
    trainer.train()


if __name__ == "__main__":
    main()