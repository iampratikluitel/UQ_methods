import register_dataset  # noqa: F401
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.WEIGHTS = "./output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # ← let everything through for COCO eval
cfg.OUTPUT_DIR = "./output"

# Build model AND load weights
model = DefaultTrainer.build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # ← actually loads your checkpoint
model.eval()

evaluator = COCOEvaluator(
    "pepper_eval",
    output_dir="./output/final_eval"
)
loader = build_detection_test_loader(cfg, "pepper_eval")

results = inference_on_dataset(model, loader, evaluator)
print(results)