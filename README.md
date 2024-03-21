# Scene To Model

## Objective
The project in question uses an Object Detection Model to predict the IoT icons used to depict an IoT scenario designed by the user. The prediction is then sent to an ADOxx server that realises the digital copy of the scenario, facilitating the prototyping and the management of complex IoT scenario.

The prediction to send consist of a JSON file with:
- an object's array like this:

  ![object_predicted](https://i.postimg.cc/WzZswync/object-predicted.png)

  Where you can find the type of the IoT device with all related arrows (useful for finding the link among the IoT devices) connected and the coordinates of the bounding box which circumscribes the icon.

- the `uuid` and `timestamp` of the image predicted:
  
  ![uuid & timestamp](https://i.postimg.cc/50RwCXPk/uuid-timestamp-json.png)



## Requirements
- Linux, macOS or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install);
- [Python3](https://www.python.org/downloads/) >= 3.7;
- [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx);
- Database in [MongoDB Atlas](https://www.mongodb.com/atlas).

In order to execute the project follow the steps below:

### Installation
- clone the repo and enter in the folder created:
  - via HTTPS
    ```
    git clone https://github.com/lollobeach/SceneToModel
    cd ./SceneToModel
    ```
  - via SSH
    ```
    git clone git@github.com:lollobeach/SceneToModel.git
    cd ./SceneToModel
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

-----

### Configuration
In adding to the installation you have to add an `.env` file in `./SceneToModel/fastapi_server/` with the following parameters:
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
refer to the parameters for the model's configuration

-----

### Run
```
cd ./fastapi_server
python3 main.py
```

At this point you can use clients like [Postman](https://www.postman.com/) to send the image to predict via HTTP POST request at the local endpoint *http://127.0.0.1:8000/uploadfile* like in the image below:
![postman image](https://i.postimg.cc/m2MCn6nk/postman.png)

> **NOTE:** the _key_ of the form data must be **image** according to the API written in the `main.py` file

With the JSON response of the predicition you can also see the image with bounding boxes predicted using a browser and going to the url *http://127.0.0.1:8000/get-image/uuid_image* where `uuid_image` is the filename (without the extension) of the picture saved in `./SceneToModel/fastapi_server/predictions/`.
