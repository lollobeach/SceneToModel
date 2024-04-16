# Object Detection

## Objective
The project in question uses an Object Detection Model to predict the IoT icons, in an image, used to depict an IoT scenario designed by the user. The prediction is then sent to an ADOxx server that realises the digital copy of the scenario, facilitating the prototyping and the management of complex IoT scenario.

The prediction to send consist of a JSON file with:
- an object's array like this:

  ![object_predicted](https://i.postimg.cc/WzZswync/object-predicted.png)

  where you can find the type of the IoT device with all related arrows (useful for finding the link among the IoT devices) connected and the coordinates of the bounding box which circumscribes the icon.

- the `uuid` and `timestamp` of the image predicted:
  
  ![uuid & timestamp](https://i.postimg.cc/50RwCXPk/uuid-timestamp-json.png)

-----

## Training and Validation
[Google Colab](https://colab.google/) was used to train the model. Starting from the documentation of **Detectron2**, some parameters were changed for this use case, like the number of classes to predict and the number of iterations to execute for the training phase.

Icons recognized:
- **Gateway**
- **Movement**
- **Temperature**
- **Arrow start 01**
- **Arrow start 02**
- **Arrow start 03**
- **Arrow start 04**
- **Arrow start 05**
- **Arrow start 06**
- **Arrow end 01**
- **Arrow end 02**
- **Arrow end 03**
- **Arrow end 04**
- **Arrow end 05**
- **Arrow end 06**

`OBJ_detectron2.ipynb` provides the Python notebook with the configured model for the training and validation phase.

### How to configure the training and the validation

1. Register the datasets:
    ```
    root_path = "/content/drive/MyDrive/BigData_Project/big_data.v20i.coco/"
    register_coco_instances("coco_training", {}, root_path + "train/_annotations.coco.json", root_path + "train/")
    register_coco_instances("coco_validation", {}, root_path + "valid/_annotations.coco.json", root_path + "valid/")
    register_coco_instances("coco_test", {}, root_path + "test/_annotations.coco.json", root_path + "test/")
    ```

2. Check the correctness of labeling:
   ```
   training_metadata = MetadataCatalog.get("coco_training")
   training_dicts = DatasetCatalog.get("coco_training")
   for d in random.sample(training_dicts, 3):
       img = cv2.imread(d["file_name"])
       visualizer = Visualizer(img[:, :, ::-1], metadata=training_metadata, scale=1)
       out = visualizer.draw_dataset_dict(d)
       cv2_imshow(out.get_image()[:, :, ::-1])
   ```

3. Set the parameters:
    ```
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("coco_training",)
    cfg.DATASETS.TEST = ("coco_validation",) # dataset for validation phase
    cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/BigData_Project/output/model_final.pth" # weights of the last training
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 5  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 4000 # iterations number for the training
    cfg.SOLVER.STEPS = [] # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 5 # used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16 # number of classes to recognize +1 since that with Roboflow we get the "super category" (the latter does not affect the training)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # after how many iterations a checkpoint of the training is saved
    cfg.TEST.EVAL_PERIOD = 500 # after how many iterations an evaluation phase is applied
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs("/content/drive/MyDrive/BigData_Project/output", exist_ok=True)
    cfg.OUTPUT_DIR = "/content/drive/MyDrive/BigData_Project/output/"
    trainer = Trainer(cfg) # custom Trainer with the validation phase implemented
    trainer.resume_or_load(resume=False) # if the last training was interrupted before its ending set "resume" to True for restarting from the last checkpoint
    trainer.train()
    ```
  
    In this case the only parameters changed were:
    - `cfg.DATASETS.TRAIN` -> is a tuple with the names of the registered datasets for the training;
    - `cfg.DATSETS.TEST` -> is a tuple with the names of the registered datasets for the validation;
    - `cfg.MODEL.WEIGHTS` -> is the path with the last `.pth` file containing the weights of the last training;
    - `cfg.SOLVER.MAX_ITER` -> is the number of iterations to apply for the training (_epochs_ = (MAX_ITER * IMS_PER_BATCH) / _sizeTrainingDataset_);
    - `cfg.MODEL.ROI_HEADS.NUM_CLASSES` -> is the number of classes to predict + 1 (Roboflow generates a "super category" not useful for the training);
    - `cfg.SOLVER.CHECKPOINT_PERIOD` -> after how many iterations a checkpoint of the training is saved;
    - `cfg.TEST.EVAL_PERIOD` -> after how many iterations an evaluation phase is applied (it returns the AP for each category to predict);
    - `cfg.OUTPUT_DIR` -> where the model saves the file with the final weights of the training;
   <br>
   
   > For other configurations follow this [link](https://detectron2.readthedocs.io/en/latest/modules/config.html)

    In addition:
    - `trainer = Trainer(cfg)` is used to develop the validation phase, where `Trainer(cfg)` is a custom trainer:
        ```
        from detectron2.data import build_detection_test_loader
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.engine import DefaultTrainer
        
        class Trainer(DefaultTrainer):
        
          @classmethod
          def build_evaluator(cls, cfg, dataset_name):
            os.makedirs("/content/drive/MyDrive/BigData_Project/inference", exist_ok=True)
            return COCOEvaluator(dataset_name, cfg, False, "/content/drive/MyDrive/BigData_Project/inference")
        
          def do_train(self):
            super().do_train()
            val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0])
            evaluator = self.build_evaluator(self.cfg, self.cfg.DATASETS.TEST[0])
            inference_on_dataset(self.model, val_loader, evaluator)
        ```
    - `trainer.resume_or_load(resume=False)` if setted to `True` allows us to restart the training from the last checkpoint if it was interrupted

4. Test the inference setting the `predictor`
     ```
     from detectron2.checkpoint import DetectionCheckpointer
     from detectron2.modeling import build_model
    
     cfg = get_cfg()
     cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16 # number of classes to recognize +1 since that with Roboflow we get the "super category" (the latter does not affect the
     training)
     cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/BigData_Project/output/model_final.pth"  # path to the model we just trained
     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
     predictor = DefaultPredictor(cfg)
     ```
     and visualizing the bounding boxes predicted
     ```
     from detectron2.utils.visualizer import ColorMode

     validation_metadata = MetadataCatalog.get("coco_validation")
    
     for d in os.listdir(root_path + "/test"):
       if ".json" not in d:
         im = cv2.imread(root_path + "/test/" + d)
         outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
         v = Visualizer(im[:, :, ::-1],
                         metadata=validation_metadata,
                         scale=0.5,
                         instance_mode=ColorMode.IMAGE   # remove the colors of unsegmented pixels. This option is only available for segmentation models
         )
         out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
         cv2_imshow(out.get_image()[:, :, ::-1])
     ```

5. At the end we will get a `model_final.pth` which can be used to apply the inference for the next images passed to the model.

   
### Dataset

The greatest effort was made in the creation of the dataset since that custom icons were used to depict the IoT devices. For this purpose [Roboflow](https://roboflow.com/) was useful, as it provides an user friendly interface to manage the labeling of the images by facilitating also teamwork. In addition, Roboflow allows to save the dataset in the format supported by this model, **COCO**, and to keep the old versions of the dataset.

The dataset used is available in `./fastapi_server/obj_detection_model/big_data.v20i.coco/` and it consists of:
- 1034 images (with augmentation applied) divided in:
  
| Training Set | Validation Set | Test Set |
| :---:        | :---:          | :---:    |  
| 951          | 44             | 39       |

- changes applied:

| Preprocessing         | Augmentation                             |
| -----                 | -----                                    |  
| Auto Orient           | Filp: Horizontal Vertical                |
| Stretch to 640x640    | Shear: ±10° Horizontal, ±10° Vertical    |

> When the _Augmentation_ is applied Roboflow allows to increment the number of images in the training set.
> 
> The original dataset contains **317** images in the training set.

- labels:
  - Arrowheads:
    
    |            | arrow_end_01 | arrow_end_02 | arrow_end_03 | arrow_end_04 | arrow_end_05 | arrow_end_06 |
    | -----      | :---:        | :---:        | :---:        | :---:        | :---:        | :---:        |
    | Training   | 453          | 486          | 447          | 462          | 489          | 486          |
    | Validation | 19           | 6            | 6            | 2            | 1            | 10           |

  - Arrow start:
    
    |            | arrow_start_01 | arrow_start_02 | arrow_start_03 | arrow_start_04 | arrow_start_05 | arrow_start_06 |
    | -----      | :---:          | :---:          | :---:          | :---:          | :---:          | :---:          |
    | Training   | 453            | 459            | 483            | 465            | 474            | 459            |
    | Validation | 16             | 5              | 9              | 4              | 3              | 6              |

  - Devices:
 
    |            | Gateway  | Temperature | Movement |
    | ----       | :---:    | :---:       | :---:    |
    | Training   | 774      | 789         | 777      | 
    | Validation | 57       | 59          | 60       |

  > As that the number of images in the training set is tripled by the augmentation with respect to the original dataset of 317 images, also the number of labels is greater than the number of labels in the original dataset.

-----

## Architecture
![architecture](https://i.postimg.cc/qqf59sS2/architecture.png)

### ADOxx
[ADOxx](https://www.adoxx.org/live/home) is a Metamodelling Platform for implementing modelling methods with its own Modelling Language Implementation. It provides also a language to develop automations.

### FastAPI
[FastAPI](https://fastapi.tiangolo.com/) is a Python framework to quickly develop APIs and it is in charge of to keep the communication among the other components: it receives the image from ADOxx to send to the Object Detection Model and the result is saved in a JSON file that can be stored in a MongoDB istance or also in the filesystem. With the JSON file the Model returns also the image with the bounding boxes predicted, also the latter can be stored in the filesystem and it is accessible via API.

### Detectron2
[Detectron2](https://ai.meta.com/tools/detectron2/) is a platform for object detection and segmentation developed by _Facebook AI Research_ and it is based on [PyTorch](https://pytorch.org/). The platform provides several configuration files to choose a pre-trained model to start with.
In this case this configuration file was enough:
```
COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
```

### MongoDB
[MongoDB](https://www.mongodb.com/) is a NoSQL Database Document Oriented that provides a local use by downloading it to their own machine or thanks to [Mongo Atlas](https://www.mongodb.com/atlas) it can be used via the cloud.
> [!NOTE]
> The connection with a Mongo DB istance on Mongo Atlas does not work with the UnicamEasyWifi. Use another internet connection.

------

## Requirements
- Linux, macOS or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install);
- [Python3](https://www.python.org/downloads/) >= 3.7;
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx);
- [CUDA](https://developer.nvidia.com/cuda-downloads);
- Database in [MongoDB Atlas](https://www.mongodb.com/atlas).

In order to execute the project follow the steps below:

### Installation

- _with WSL may be useful to execute_:
  ```
  sudo apt-get update
  sudo apt-get install gcc
  sudo apt-get install g++
  sudo apt-get install python3-dev
  sudo apt install python3-venv
  sudo apt-get install libgl1-mesa-dev
  ```
- clone the repo and enter in the folder created:
  - via HTTPS
    ```
    git clone https://github.com/PROSLab/TBDM-VGLS-2023.git
    cd ./TBDM-VGLS-2023/ObjectDetection
    ```
  - via SSH
    ```
    git clone git@github.com:PROSLab/TBDM-VGLS-2023.git
    cd ./TBDM-VGLS-2023/ObjectDetection
    ```
- create the virtual environment and execute it:
  ```
  python3 -m venv venv
  source venv/bin/activate
  ```
- install the requirements **(follow the order in the snippet)**:
  ```
  pip install -r requirements.txt
  pip install -r git_requirement.txt
  ```
The first file, _requirements.txt_, contains all libraries used for the backend and the model, while the second file, _git_requirements.txt_, contains the github repository of the Detectron2 model.

### Configuration
In addition to the installation you have to add an `.env` file in `./TBDM-VGLS-2023/ObjectDetection/fastapi_server/` with the following parameters:
```
CONNECTION_STRING=mongodb+srv://<user>:<password>@cluster0.ictsbaw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DATABASE_NAME=<database_name>
COLLECTION_NAME=<collection_name>
DATASET_PATH=./obj_detection_model/big_data.v20i.coco/
NUM_CLASSES=16
SCORE_THRESH_TEST=0.7
MODEL_WEIGHTS=./obj_detection_model/model_final.pth
```

The first three lines:
```
CONNECTION_STRING=mongodb+srv://<user>:<password>@cluster0.ictsbaw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
DATABASE_NAME=<database_name>
COLLECTION_NAME=<collection_name>
```
refer to a Mongo Atlas istance but it does not prohibit using MongoDB locally.

The last ones:
```
DATASET_PATH=./obj_detection_model/big_data.v20i.coco/
NUM_CLASSES=16
SCORE_THRESH_TEST=0.7
MODEL_WEIGHTS=./obj_detection_model/model_final.pth
```
refer to the parameters for the model's configuration, where:
- **DATASET_PATH** corresponds to the path of the dataset
- **NUM_CLASSES** corresponds to the number of categories to recognise (3 for the IoT devices, 12 for the end and start arrows, 1 is a category created by Roboflow not used by the model)
- **SCORE_THRESH_TEST** corresponds to the threshold for which if the prediction accuracy does not exceed it then the model does not return the bounding box
- **MODEL_WEIGHTS** corresponds to the path of the file with the weights of the last training

-----

## Run
```
cd ./fastapi_server
python3 main.py
```

At this point you can use a client like [Postman](https://www.postman.com/) to send the image to predict via HTTP POST request at the local endpoint */uploadfile* like in the image below:
![postman image](https://i.postimg.cc/m2MCn6nk/postman.png)
In `./testimages` you can find some picture to use for doing some test.

> [!NOTE]
> the _key_ of the form data must be **image** according to the API written in the `main.py` file.

In addition to the JSON response of the predicition you can also see the image with bounding boxes predicted using a browser and going to the url */get-image/uuid_image* where `uuid_image` is the filename (without the extension) of the picture saved in `./TBDM-VGLS-2023/ObjectDetection/fastapi_server/predictions/`.

For example:

![image predicted](https://i.postimg.cc/vTRrQ3QH/example.png)
