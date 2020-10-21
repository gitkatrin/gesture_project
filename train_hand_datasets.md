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

