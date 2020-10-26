# How to train hand datasets with the SSD with Mobilenet v2

- [1. File structure](https://github.com/gitkatrin/gesture_project/blob/master/train_hand_datasets.md#file-structure)
  - [1.1 Model folder from Tensorflow]()
  - [1.2 Workspace and Scripts folder]()
- [2. Preparing settings]()
  - [2.1 Software]()
  - [2.2 Preparing the data]()
    - [2.2.1 Generate .tfrecord file from .csv file]()
- [Training]()
  - [Train the Model](l)
  
# 1. File structure:

## 1.1 Model folder from Tensorflow
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

## 1.2 Workspace and Scripts folders
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

# 2. Preparing settings

## 2.1 Software
- Tensorflow, Tensorflow-gpu
- Python 3
- [Tensorflow models](https://github.com/tensorflow/models)
- Python environement (```python3 -m venv --system-site-packages ./venv```)
  - ```pip3 install protobuf```
- [scripts for data preparation](https://github.com/gitkatrin/gesture_project/tree/master/scripts)
- [Pre-trained Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)


## 2.2 Preparing files

1. create your [label map](https://github.com/gitkatrin/gesture_project/blob/master/scripts/label_map.pbtxt) here ```direction_folder/workspace/training_demo/annotations```


### 2.2.1 Generate .tfrecord-file from .csv-file

1. activate the Python environement with ```source ./venv/bin/activate``` (in terminal)
2. go to the research folder in models with ```cd models/reseach```
3. compile the protos with 
  ```
  protoc object_detection/protos/*.protos --python_out=.
  export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
  sudo python3 setup.py install
  ```
4. open the [csv_to_tfrecord.py](https://github.com/gitkatrin/gesture_project/blob/master/scripts/csv_to_tfrecord.py) file
5. change the path directories in the main function for:
  - your image path
  - your csv path
  - your output path
  - your label map path
6. go back into the terminal and go to path ```cd workspace/training_demo``` 
7. run the script with ```python3 csv_to_tfrecord.py```

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# 3. Training

## 3.1 Train the Model

in the **virtual environment**, in file  ```cd hand_detection_egohands/workspace/training_demo``` use the following command:
```
python3 model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models/my_ssd_mobilenet_v2_fpnlite/my_ssd_mobilenet_v2.config
```
At the end the **loss should be between 0.150 and 0.200** to prevents unnderfitting and overfitting.


## 3.2 Monotoring realtime training (optional)

It is possible to show the traning process with TensorBoard. This is optional.

in the **virtual environment**, in file  ```cd hand_detection_egohands/workspace/training_demo``` use the following command:
```
tensorboard --logdir=models/my_ssd_mobilenet_v2_fpnlite
```
It should open an URL-Link to the TensorBoard Server. Open this link in your web browser and you can continuously monitor training.


## 3.3 Exporting the Inference Graph

in the **virtual environment**, in file  ```cd hand_detection_egohands/workspace/training_demo``` use the following command:
```
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_mobilenet_v2_fpnlite/my_ssd_mobilenet_v2.config --trained_checkpoint_dir ./models/my_ssd_mobilenet_v2_fpnlite/ --output_directory ./exported-models/my_mobilenet_model
```

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# Evaluation

in the **virtual environment**, in file  ```cd hand_detection_egohands/workspace/training_demo``` use the following command:
```
python3 model_main_tf2.py --pipeline_config_path models/my_ssd_mobilenet_v2_fpnlite/ssd_mobilenet_v2.config --model_dir models/my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models/my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

[comment]: # (---------------------------------------------------------------------------------------------------------------------------------------------------------------)

# Testing

in the **virtual environment**, in file  ```cd hand_detection_egohands/workspace/training_demo``` use the following command:
```
python3 TF-image-od.py
```
There are also some other scripts for detection with different input:
- for detecting and coundting objects on an **image**: ```python3 TF-image-object-counting.py```
- for detecting objects in a **video**: ```python3 TF-video-od.py```
- for detecting and counting objects in a **video**: ```python3 TF-video-object-counting.py```
- for detection objects live on **webcam**: ```python3 TF-webcam-opencv.py```

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
