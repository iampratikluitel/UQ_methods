import cv2
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def visualize_gt_vs_pred(sample, predictor, metadata, save_path):
    image = cv2.imread(sample["file_name"])

    # Ground truth
    from detectron2.data import detection_utils as utils
    vis_gt = Visualizer(image[:, :, ::-1], metadata=metadata)
    gt_instances = utils.annotations_to_instances(sample["annotations"], image.shape[:2])
    vis_gt_out = vis_gt.overlay_instances(
        boxes=gt_instances.gt_boxes,
        labels=[metadata.thing_classes[c] for c in gt_instances.gt_classes],
        masks=gt_instances.gt_masks if gt_instances.has("gt_masks") else None,
    )

    # Prediction
    outputs = predictor(image)
    vis_pred = Visualizer(image[:, :, ::-1], metadata=metadata)
    vis_pred_out = vis_pred.draw_instance_predictions(outputs["instances"].to("cpu"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(vis_gt_out.get_image())
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(vis_pred_out.get_image())
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()