from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from PIL import Image
import json
import numpy as np
import dotenv
import os
from pymongo import MongoClient
from uuid import uuid4
import time

dotenv.load_dotenv()

mongo_client = MongoClient(os.getenv("CONNECTION_STRING"))
database = mongo_client[os.getenv("DATABASE_NAME")]
collection = database[os.getenv("COLLECTION_NAME")]


def get_annotations():
    with open("./obj_detection_model/_annotations.coco.json", "r") as f:
        return json.load(f)


def dataset_registration():

    categories = map(lambda x: x["name"], get_annotations()["categories"])

    if "coco_training" in DatasetCatalog.list():
        DatasetCatalog.remove("coco_training")
    if "coco_validation" in DatasetCatalog.list():
        DatasetCatalog.remove("coco_validation")

    root_path = os.getenv("DATASET_PATH")
    register_coco_instances("coco_training", {}, root_path +
                            "train/_annotations.coco.json", root_path + "train/")
    register_coco_instances("coco_validation", {}, root_path +
                            "train/_annotations.coco.json", root_path + "valid/")

    MetadataCatalog.get("coco_validation").set(thing_classes=list(categories))
    validation_metadata = MetadataCatalog.get("coco_validation")
    # training_metadata = MetadataCatalog.get("coco_training")

    return validation_metadata


def euclidean_distance(point1, point2):
    array_1, array_2 = np.array(point1), np.array(point2)
    squared_distance = np.sum(np.square(array_1 - array_2))
    distance = np.sqrt(squared_distance)
    return distance


def arrow_connections(info, category, next_index, bbox, bounding_boxes):
    if euclidean_distance((bbox[0].item(), bbox[1].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[1].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[3].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[3].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[1].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[1].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[1].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[1].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[1].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[1].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[3].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[3].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[0].item(), bbox[3].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[3].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][1].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[3].item()), (bounding_boxes[next_index][0].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True
    elif euclidean_distance((bbox[2].item(), bbox[3].item()), (bounding_boxes[next_index][2].item(), bounding_boxes[next_index][3].item())) < 80:
        info["directions"].append(category["name"])
        return True


def json_writing(bounding_boxes, pred_classes, uuid):
    # how to retrieve items from torch tensor object:
    # x = torch.tensor([[1,2,3][4,5,6]]) -> x[1][2] == 6

    results = {
        "predictions": []
    }

    for index, bbox in enumerate(bounding_boxes):
        info = {}
        for category in get_annotations()["categories"]:
            if pred_classes[index] == category["id"]:
                if not "arrow" in category["name"]:
                    info["category"] = category["name"]
                    info["directions"] = []

                    for new_index in range(0, len(bounding_boxes)):
                        for category in get_annotations()["categories"]:
                            if pred_classes[new_index] == category["id"]:
                                if "arrow" in category["name"]:
                                    if arrow_connections(info, category, new_index, bbox, bounding_boxes):
                                        break

                    break
                else:
                    info["category"] = category["name"]

        if "category" in info:
            info["x"] = bbox[0].item()
            info["y"] = bbox[1].item()
            info["w"] = bbox[2].item()
            info["h"] = bbox[3].item()
            results["predictions"].append(info)

        results["uuid"] = uuid
        results["timestamp"] = int(time.time())

    response = results.copy()
    json_object = json.dumps(results)
    collection.insert_one(results)

    with open("./predictions/" + uuid + ".json", "w") as json_file:
        json_file.write(json_object)

    return response


def inference(im):

    metadata = dataset_registration()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(os.getenv("NUM_CLASSES"))
    cfg.MODEL.WEIGHTS = os.getenv("MODEL_WEIGHTS")
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(
        os.getenv("SCORE_THRESH_TEST"))   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    outputs = predictor(im)
    # bboxs format: x,y,w,h (x,y are the coordinates of the top-left corner)
    instances = outputs["instances"]
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata,
                   scale=1,
                   instance_mode=ColorMode.IMAGE
                   )
    out = v.draw_instance_predictions(instances.to("cpu"))
    img = Image.fromarray(out.get_image()[:, :, ::-1])

    uuid = str(uuid4())
    image_name = uuid + ".jpg"
    img.save("./predictions/" + image_name)

    bounding_boxes = instances._fields["pred_boxes"].tensor  # torch_tensor
    pred_classes = instances._fields["pred_classes"]  # torch_tensor

    return json_writing(bounding_boxes, pred_classes, uuid)
