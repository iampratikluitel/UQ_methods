from detectron2.data import MetadataCatalog
import register_dataset
meta = MetadataCatalog.get("pepper_train")
print(meta)
print("thing_classes:", meta.thing_classes)