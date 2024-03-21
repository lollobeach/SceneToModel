# Scene To Model

## Requirements
- Linux, macOS or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install);
- [Python3](https://www.python.org/downloads/) >= 3.7

In order to execute the project follow the steps below:
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
- install the requirements (follow the order in the snippet):
  ```
  pip install -r requirements.txt
  pip install -r git_requirements.txt
  ```
The first file, _requirements.txt_, contains all libraries used for the backend and the model, while the second file, _git_requirements.txt_, containes the github repository of the Detectron2 mode. 
