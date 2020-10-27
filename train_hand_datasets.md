# How to train hand datasets with the SSD with Mobilenet v2

- [1. File Structure](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#1-file-structure)
  - [1.1 Model Folder from Tensorflow](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#11-model-folder-from-tensorflow)
  - [1.2 Workspace and Scripts Folders](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#12-workspace-and-scripts-folders)
- [2. Preparing Settings](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#2-preparing-settings)
  - [2.1 Software](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#21-software)
  - [2.2 Preparing Files](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#22-preparing-files)
    - [2.2.1 Create Label Map File](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#221-create-label-map-file)
    - [2.2.2 Generate .tfrecord-File from .csv-File](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#222-generate-tfrecord-file-from-csv-file)
    - [2.2.3 Change Configuration File](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#223-change-configuration-file)
- [3. Training](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#3-training)
  - [3.1 Train the Model](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#31-train-the-model)
  - [3.2 Monitoring realtime Training (optional)](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#32-monotoring-realtime-training-optional)
  - [3.3 Exporting the Interference Graph](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#33-exporting-the-inference-graph)
- [4. Evaluation](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#4-evaluation)
  - [4.1 Evaluate the Model](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#41-evaluate-the-model)
  - [4.2 Monotoring realtime Evaluation (optional)](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#42-monotoring-realtime-evaluation-optional)
- [5. Testing](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#5-testing)
  
# 1. File Structure

## 1.1 Model Folder from Tensorflow
```
detection_folder/
|-- models                  # from tensorflow https://github.com/tensorflow/models
|   |-- ...
|   |-- research
|   |   |-- ...
|   |   |-- object_detection
|   |   |   |-- ...
|   |   |   |-- configs     # configurations for all TF2 models
|   |   |   |-- ...
|   |   |   |-- models      # all usable models
|   |   |   |-- packages    # setup files for tf1 and tf2
|   |   |   |-- predictors  # all usable predictors
|   |   |   |-- ...
|   |   |   |-- samples     # some usable configuration samples
|   |   |   |-- ...
|   |   |-- ...
|   |   |-- model_main_tf2.py
|   |   |-- README.md
|   |   `-- setup.py
|   `-- ...
```

## 1.2 Workspace and Scripts Folders
```
detection_folder/
|-- ...
|-- scripts                   # helpful scripts for data preparation (in this repository: https://github.com/gitkatrin/gesture_project/tree/master/scripts)
|-- workspace
|   `-- training_demo
|       |-- annotations
|       |   |-- label_map.pbtxt
|       |   |-- test.record
|       |   `-- train.record
|       |-- exported-models   # this is empty at the beginning of training, your new trained model will be here
|       |-- images
|       |   |-- test
|       |   |-- train
|       |-- models            # model which you use for training
|       |   `-- my_ssd_mobilenet_v2_fpnlite
|       |       `-- my_ssd_mobilenet_v2.config   # copy .config from pre-trained model
|       |-- pre-trained-models                   # pre-trained model which you use for training (download it here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
|       |   `-- ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8
|       |       |-- checkpoint
|       |       |-- saved_model
|       |       `-- pipeline.config              # configuration file
|       |-- exporter_main_v2.py
|       |-- model_main_tf2.py
|       |-- TF-image-object-counting.py
|       |-- TF-image-od.py
|       |-- TF-video-object-counting.py
|       |-- TF-video-od.py
|       `-- TF-webcam-opencv.py
`-- Training-a-Custom-TensorFlow-2.X-Object-Detector-master.zip
```

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# 2. Preparing Settings

## 2.1 Software
- Tensorflow, Tensorflow-gpu
- Python 3
- [Tensorflow models](https://github.com/tensorflow/models)
- Python environement (```python3 -m venv --system-site-packages ./venv```)
  - ```pip3 install protobuf```
- [scripts for data preparation](https://github.com/gitkatrin/gesture_project/tree/master/scripts)
- [Pre-trained Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) (in this case: SSD MobileNet V2 FPNLite 640x640)
- Datasets: [Oxford Dataset](https://www.robots.ox.ac.uk/~vgg/data/hands/), [Egohands Dataset](http://vision.soic.indiana.edu/projects/egohands/)


## 2.2 Preparing Files

### 2.2.1 Create Label Map File

1. create a file named ```label_map.pbtxt``` in this folder: ```direction_folder/workspace/training_demo/annotations```
2. open the file and identify your items that you would like to detect. It should look like this:
    ```
    item {
    id: 1
    name: 'hand'
    }
    item {
    id: 2
    name: 'face'
    }
    ```
    the file for hand detection is [here](https://github.com/gitkatrin/gesture_project/blob/master/scripts/label_map.pbtxt)

### 2.2.2 Generate .tfrecord-File from .csv-File

1. go to the research folder in models with ```cd models/reseach``` without virtual environment
2. compile the protos with:
    ```
    protoc object_detection/protos/*.proto --python_out=.
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    sudo python3 setup.py install
    ```
3. open the [csv_to_tfrecord.py](https://github.com/gitkatrin/gesture_project/blob/master/scripts/csv_to_tfrecord.py) file
4. change the path directories in the main function for:
    - Line 72: your image path
    - Line 73: your csv path
    - Line 74: your output path
    - Line 75: your label map path
5. go back into the terminal and go to path ```cd workspace/training_demo``` 
6. run the script with ```python3 csv_to_tfrecord.py```

### 2.2.3 Change Configuration File

1. copy the configuration file ```detection_folder/workspace/training_demo/pre-trained-model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config``` to ```detection_folder/workspace/training_demo/models/my_ssd_mobilenet_v2_fpnlite``` folder
2. open the configuration file in ```detection_folder/workspace/training_demo/models/my_ssd_mobilenet_v2_fpnlite``` folder
3. do the following changes:
    - Line 3:   change ```num_classes``` to this value of different objects you want to detect
    - Line 135: change ```batch_size``` to 4
    - Line 165: change ```fine_tune_checkpoint``` to your path (```pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0```)
    - Line 171: change ```fine_tune_checkpoint_type``` to ```detection```
    - Line 175: change ```label_map_path``` to your path (```annotation/label_map.pbtxt```)
    - Line 177: change ```input_path``` to your path (```annotation/train.record```)
    - Line 185: change ```label_map_path``` to your path (```annotation/label_map.pbtxt```)
    - Line 189: change ```input_path``` to your path (```annotation/test.record```)

    (keep in mind: if you choose a different model, than the line numbers will be different)
    
[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# 3. Training

## 3.1 Train the Model

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
python3 model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models/my_ssd_mobilenet_v2_fpnlite/my_ssd_mobilenet_v2.config
```
At the end the **loss should be between 0.150 and 0.200** to prevents unnderfitting and overfitting.


## 3.2 Monotoring realtime Training (optional)

It is possible to show the traning process with TensorBoard. This is optional.

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
tensorboard --logdir=models/my_ssd_mobilenet_v2_fpnlite
```
It should open an URL-Link to the TensorBoard Server. Open this link in your web browser and you can continuously monitor training.


## 3.3 Exporting the Inference Graph

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_mobilenet_v2_fpnlite/my_ssd_mobilenet_v2.config --trained_checkpoint_dir ./models/my_ssd_mobilenet_v2_fpnlite/ --output_directory ./exported-models/my_mobilenet_model
```

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# 4. Evaluation

## 4.1 Evaluate the Model

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
python3 model_main_tf2.py --pipeline_config_path models/my_ssd_mobilenet_v2_fpnlite/my_ssd_mobilenet_v2.config --model_dir models/my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models/my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

## 4.2 Monotoring realtime Evaluation (optional)

It is possible to show the evaluation process with TensorBoard. This is optional.

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
tensorboard --logdir=models/my_ssd_mobilenet_v2_fpnlite
```
It should open an URL-Link to the TensorBoard Server. Open this link in your web browser and you can continuously monitor evaluation.

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# 5. Testing

in the **virtual environment**, in file  ```cd detection_folder/workspace/training_demo``` use the following command:
```
python3 TF-image-od.py
```
There are also some other scripts for detection with different input:
- for detecting and coundting objects on an **image**: ```python3 TF-image-object-counting.py```
- for detecting objects in a **video**: ```python3 TF-video-od.py```
- for detecting and counting objects in a **video**: ```python3 TF-video-object-counting.py```
- for detecting objects live on **webcam**: ```python3 TF-webcam-opencv.py```
- for detecting objects live on **special webcam**: ```python3 TF-webcam-OV580.py``` 

You can also run this script with different arguments:
```
python3 TF-image-od.py [-h] [--model MODEL] [--labels LABELS] [--image IMAGE] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Folder that the Saved Model is Located In
  --labels LABELS       Where the Labelmap is Located
  --image IMAGE         Name of the single image to perform detection on
  --threshold THRESHOLD Minimum confidence threshold for displaying detected objects
```
