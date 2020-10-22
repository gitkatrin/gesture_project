# How to train hand datasets with the SSD with Mobilenet v1


# File structure:
```
hand_detection_egohand/
|-- models # from tensorflow https://github.com/tensorflow/models
|   |-- ...
|   |-- research
|   |   |-- ...
|   |   |-- annotations   ???????????????
|   |   |   |-- label_map.pbtxt
|   |   |   |-- test.record
|   |   |   `-- train.record
|   |   |-- images ?????????????????????????
|   |   |-- models  ??????????????????????
|   |   |   `-- my_ssd_mobilenet_v2_fpnlite
|   |   |       `-- pipeline.config
|   |   |-- object_detection
|   |   |   |-- ...
|   |   |   |-- annotations ????????????????????????????????????????????????????????????
|   |   |   |   |-- label_map.pbtxt
|   |   |   |   |-- test.record
|   |   |   |   `-- train.record
|   |   |   |-- checkpoint ??????????????????????????????????????????????????????
|   |   |   |   |-- checkpoint
|   |   |   |   |-- ckpt-0.data-00000-of-00001
|   |   |   |   `-- ckpt-0.index
|   |   |   |-- configs   # configurations for all TF2 models
|   |   |   |-- ...
|   |   |   |-- models  # all usable models
|   |   |   |-- packages  # setup files for tf1 and tf2
|   |   |   |-- predictors  # all usable predictors
|   |   |   |-- ...
|   |   |   |-- samples # some usable configuration samples
|   |   |   |-- ...
|   |   |-- pre-trained-models ???????????????????????????????????
|   |   |-- ...
|   |   |-- model_main_tf2.py
|   |   |-- README.md
|   |   `-- setup.py
|   `-- ...
```

```
hand_detection_egohand/
|-- ...
|-- scripts # helpful scripts for data prepatration (in this repository: https://github.com/gitkatrin/gesture_project/tree/master/scripts)
|-- workspace
|   `-- training_demo
|       |-- annotations
|       |   |-- label_map.pbtxt
|       |   |-- test.record
|       |   `-- train.record
|       |-- checkpoint ??????????????????????????????????????????
|       |   |-- checkpoint
|       |   |-- ckpt-0.data-00000-of-00001
|       |   `-- ckpt-0.index
|       |-- exported-models # this is empty at the beginning of training, your new trained model will be here
|       |-- images
|       |   |-- test
|       |   |-- train
|       |-- models # model which you use for training (download it here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
|       |   `-- my_ssd_mobilenet_v2_fpnlite
|       |       `-- pipeline.config # copy from pre-trained model
|       |-- pre-trained-models      # pre-trained model which you use for training (download it here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
|       |   `-- ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8
|       |       |-- checkpoint
|       |       |-- saved_model
|       |       `-- pipeline.config # configuration file
|       |-- exporter_main_v2.py
|       |-- model_main_tf2.py
|       |-- TF-image-object-counting.py
|       |-- TF-image-od.py
|       |-- TF-video-object-counting.py
|       |-- TF-video-od.py
|       `-- TF-webcam-opencv.py
`-- Training-a-Custom-TensorFlow-2.X-Object-Detector-master.zip
```

# Training
## Exporting the Inference Graph
in the Venv, in file:
```
cd hand_detection_egohands/workspace/training_demo
```
Use the following command:
```
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/my_ssd_mobilenet_v2_fpnlite/ssd_mobilenet_v2.config --trained_checkpoint_dir ./models/my_ssd_mobilenet_v2_fpnlite/ --output_directory ./exported-models/my_mobilenet_model
```

# Evaluation
in the Venv, in file:
```
cd hand_detection_egohands/workspace/training_demo
```
Use the following command:
```
python3 model_main_tf2.py --pipeline_config_path models/my_ssd_mobilenet_v2_fpnlite/ssd_mobilenet_v2.config --model_dir models/my_ssd_mobilenet_v2_fpnlite --checkpoint_dir models/my_ssd_mobilenet_v2_fpnlite --alsologtostderr
```

# Testing
in the Venv, in file:
```
cd hand_detection_egohands/workspace/training_demo
```
Use the following command for detecting objects on an image:
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
