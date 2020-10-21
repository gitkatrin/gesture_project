# How to train hand datasets with the SSD with Mobilenet v1


# File structure:

```
project_name/
├─ models/
|  ├─ community/
|  ├─ official/
|  ├─ orbit/
|  ├─ research/
├─ scripts/
├─ workspace/
|  └─ training_demo/
|     ├─ annotations/
|     ├─ exported-models/
|     ├─ images/
|       ├─ test/
|       └─ train/
|     ├─ models/
      ├─ pre-trained-models/
      │  └─ ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
      │     ├─ checkpoint/
      │     ├─ saved_model/
      │     └─ pipeline.config
```


```
hand_detection_egohand/
|-- models
|   |-- community
|   |-- official
|   |-- orbit
|   |-- research
|   |   |-- a3c_blogpost
|   |   |-- adversarial_text
|   |   |-- annotations   ???????????????
|   |   |   |-- label_map.pbtxt
|   |   |   |-- test.record
|   |   |   `-- train.record
|   |   |-- attention_ocr
|   |   |-- audioset
|   |   |-- autoaugment
|   |   |-- cognitive_planning
|   |   |-- cvt_text
|   |   |-- deeplab
|   |   |-- deep_speech
|   |   |-- delf
|   |   |-- efficient-hrl
|   |   |-- images ?????????????????????????
|   |   |-- lfads
|   |   |-- lstm_object_detection
|   |   |-- marco
|   |   |-- models  ??????????????????????
|   |   |   `-- my_ssd_mobilenet_v2_fpnlite
|   |   |       `-- pipeline.config
|   |   |-- nst_blogpost
|   |   |-- object_detection
|   |   |   |-- anchor_generators
|   |   |   |-- annotations ????????????????????????????????????????????????????????????
|   |   |   |   |-- label_map.pbtxt
|   |   |   |   |-- test.record
|   |   |   |   `-- train.record
|   |   |   |-- box_coders
|   |   |   |-- builders
|   |   |   |-- checkpoint ??????????????????????????????????????????????????????
|   |   |   |   |-- checkpoint
|   |   |   |   |-- ckpt-0.data-00000-of-00001
|   |   |   |   `-- ckpt-0.index
|   |   |   |-- colab_tutorials
|   |   |   |-- configs   # configurations for all TF2 models
|   |   |   |-- core
|   |   |   |-- data
|   |   |   |-- data_decoders
|   |   |   |   |-- __init__.py
|   |   |   |-- dataset_tools
|   |   |   |-- dockerfiles
|   |   |   |-- g3doc
|   |   |   |-- inference
|   |   |   |-- legacy
|   |   |   |-- matchers
|   |   |   |-- meta_architectures
|   |   |   |-- metrics
|   |   |   |-- models  # all usable models
|   |   |   |-- packages  # setup files for tf1 and tf2
|   |   |   |-- predictors  # all usable predictors
|   |   |   |-- protos
|   |   |   |-- __pycache__
|   |   |   |-- samples # some usable configuration samples
|   |   |   |-- test_data
|   |   |   |-- test_images
|   |   |   |-- tpu_exporters
|   |   |   |-- utils
|   |   |-- pre-trained-models ???????????????????????????????????
|   |   |-- rebar
|   |   |-- sequence_projection
|   |   |-- slim
|   |   |-- vid2depth
|   |   |   |-- dataset
|   |   |   |   |-- kitti
|   |   |   |   |   |-- static_frames.txt
|   |   |   |   |   |-- test_files_eigen.txt
|   |   |   |   |   |-- test_files_stereo.txt
|   |   |   |   |   |-- test_scenes_eigen.txt
|   |   |   |   |   `-- test_scenes_stereo.txt
|   |   |   |   |-- dataset_loader.py
|   |   |   |   |-- gen_data.py
|   |   |   |   `-- __init__.py
|   |   |   |-- ops
|   |   |   |   |-- testdata
|   |   |   |   |   `-- pointcloud.npy
|   |   |   |   |-- BUILD
|   |   |   |   |-- icp_grad.py
|   |   |   |   |-- icp_grad_test.py
|   |   |   |   |-- icp_op_kernel.cc
|   |   |   |   |-- icp_op.py
|   |   |   |   |-- icp_test.py
|   |   |   |   |-- icp_train_demo.py
|   |   |   |   |-- icp_util.py
|   |   |   |   |-- __init__.py
|   |   |   |   `-- pcl_demo.cc
|   |   |   |-- third_party
|   |   |   |   |-- BUILD
|   |   |   |   |-- eigen.BUILD
|   |   |   |   |-- flann.BUILD
|   |   |   |   |-- hdf5.BUILD
|   |   |   |   `-- pcl.BUILD
|   |   |   |-- BUILD
|   |   |   |-- inference.py
|   |   |   |-- model.py
|   |   |   |-- nets.py
|   |   |   |-- project.py
|   |   |   |-- reader.py
|   |   |   |-- README.md
|   |   |   |-- repo.bzl
|   |   |   |-- train.py
|   |   |   |-- util.py
|   |   |   `-- WORKSPACE
|   |   |-- model_main_tf2.py
|   |   |-- README.md
|   |   `-- setup.py
|   |-- AUTHORS
|   |-- CODEOWNERS
|   |-- CONTRIBUTING.md
|   |-- ISSUES.md
|   |-- LICENSE
|   `-- README.md
|-- scripts
|   `-- preprocessing
|       |-- generate_tfrecord.py
|       `-- partition_dataset.py
|-- workspace
|   `-- training_demo
|       |-- annotations
|       |   |-- label_map.pbtxt
|       |   |-- test.record
|       |   `-- train.record
|       |-- checkpoint
|       |   |-- checkpoint
|       |   |-- ckpt-0.data-00000-of-00001
|       |   `-- ckpt-0.index
|       |-- exported-models
|       |   `-- my_mobilenet_model
|       |       |-- checkpoint
|       |       |   |-- checkpoint
|       |       |   |-- ckpt-0.data-00000-of-00001
|       |       |   `-- ckpt-0.index
|       |       |-- saved_model
|       |       |   |-- assets
|       |       |   |-- variables
|       |       |   |   |-- variables.data-00000-of-00001
|       |       |   |   `-- variables.index
|       |       |   |-- label_map.pbtxt
|       |       |   `-- saved_model.pb
|       |       `-- pipeline.config
|       |-- images
|       |   |-- test
|       |   |   |-- i-038215c28fdc4e55be33235cf3bf8fcd.jpg
|       |   |   |-- i-038215c28fdc4e55be33235cf3bf8fcd.xml
|       |   |   |-- i-07d09c281bd94e88af9a4e3811e976e4.jpg
|       |   |   |-- i-07d09c281bd94e88af9a4e3811e976e4.xml
|       |   |   |-- i-14370c36306d44cfbb37232f766e0365.jpg
|       |   |   |-- i-14370c36306d44cfbb37232f766e0365.xml
|       |   |   |-- i-1e092ec6eabf47f9b85795a9e069181b.jpg
|       |   |   |-- i-1e092ec6eabf47f9b85795a9e069181b.xml
|       |   |   |-- i-29c86b783ec244acba67b45d5f16ded1.jpg
|       |   |   |-- i-29c86b783ec244acba67b45d5f16ded1.xml
|       |   |   |-- i-3e7d87c0bc154f93a8d939439277fafc.jpg
|       |   |   |-- i-3e7d87c0bc154f93a8d939439277fafc.xml
|       |   |   |-- i-98d05f7a289847b7b166272d70025f5f.jpg
|       |   |   |-- i-98d05f7a289847b7b166272d70025f5f.xml
|       |   |   |-- i-aec5aed72f4d43fab8957c948eb625d3.jpg
|       |   |   |-- i-aec5aed72f4d43fab8957c948eb625d3.xml
|       |   |   |-- i-c55a624481f64660a4a6721f6e1ba765.jpg
|       |   |   |-- i-c55a624481f64660a4a6721f6e1ba765.xml
|       |   |   |-- i-fbc3c593f37147b1b90920ca7e5e5af3.jpg
|       |   |   `-- i-fbc3c593f37147b1b90920ca7e5e5af3.xml
|       |   |-- train
|       |   |   |-- 004060365.jpg
|       |   |   |-- 004060365.xml
|       |   |   |-- 005730150.jpg
|       |   |   |-- 005730150.xml
|       |   |   |-- 005910853.jpg
|       |   |   |-- 005910853.xml
|       |   |   |-- 006030263.jpg
|       |   |   |-- 006030263.xml
|       |   |   |-- 00603-2540-21_45042291.jpg
|       |   |   |-- 00603-2540-21_45042291.xml
|       |   |   |-- 009041982.jpg
|       |   |   |-- 009041982.xml
|       |   |   |-- 009047915.jpg
|       |   |   |-- 009047915.xml
|       |   |   |-- 1559966609945_479678-2.jpg
|       |   |   |-- 1559966609945_479678-2.xml
|       |   |   |-- 1569555463118_660457-2.jpg
|       |   |   |-- 1569555463118_660457-2.xml
|       |   |   |-- 1579894751203_432313-2.png
|       |   |   |-- 1579894751203_432313-2.xml
|       |   |   |-- 1c84d1d5-2318-5f9b-e054-00144ff88e88.jpg
|       |   |   |-- 1c84d1d5-2318-5f9b-e054-00144ff88e88.xml
|       |   |   |-- 23171_145662686409999.jpg
|       |   |   |-- 23171_145662686409999.xml
|       |   |   |-- 2671_1421355374996996-2.png
|       |   |   |-- 2671_1421355374996996-2.xml
|       |   |   |-- 272f18b6-1b10-5621-e054-00144ff8d46c.jpg
|       |   |   |-- 272f18b6-1b10-5621-e054-00144ff8d46c.xml
|       |   |   |-- 30861_1486141826753753-2.jpg
|       |   |   |-- 30861_1486141826753753-2.xml
|       |   |   |-- 37981_1510556368140140-2.jpg
|       |   |   |-- 37981_1510556368140140-2.xml
|       |   |   |-- 421950145.jpg
|       |   |   |-- 421950145.xml
|       |   |   |-- 433530324.jpg
|       |   |   |-- 433530324.xml
|       |   |   |-- 494b88fa-a2bd-4835-e054-00144ff8d46c.jpg
|       |   |   |-- 494b88fa-a2bd-4835-e054-00144ff8d46c.xml
|       |   |   |-- 502680408.jpg
|       |   |   |-- 502680408.xml
|       |   |   |-- 50580-0496-60_FF03FFBF.jpg
|       |   |   |-- 50580-0496-60_FF03FFBF.xml
|       |   |   |-- 505800501.jpg
|       |   |   |-- 505800501.xml
|       |   |   |-- 50844-0291-08_RXNAVIMAGE10_6335B1AD.jpg
|       |   |   |-- 50844-0291-08_RXNAVIMAGE10_6335B1AD.xml
|       |   |   |-- 525440161.jpg
|       |   |   |-- 525440161.xml
|       |   |   |-- 53746-0110-05_RXNAVIMAGE10_E315F1BF.jpg
|       |   |   |-- 53746-0110-05_RXNAVIMAGE10_E315F1BF.xml
|       |   |   |-- 54321_1581752549101101 (1).png
|       |   |   |-- 54321_1581752549101101 (1).xml
|       |   |   |-- 54321_1581752549101101.png
|       |   |   |-- 54321_1581752549101101.xml
|       |   |   |-- 57896-0101-01_RXNAVIMAGE10_FB03FDAF.jpg
|       |   |   |-- 57896-0101-01_RXNAVIMAGE10_FB03FDAF.xml
|       |   |   |-- 5mg-325mg_Hydrocodone-APAP_Tablet.jpg
|       |   |   |-- 5mg-325mg_Hydrocodone-APAP_Tablet.xml
|       |   |   |-- 648_pd1738885_1.jpg
|       |   |   |-- 648_pd1738885_1.xml
|       |   |   |-- 7894bddb-4e7b-4f70-e053-2a91aa0a0cfe.jpg
|       |   |   |-- 7894bddb-4e7b-4f70-e053-2a91aa0a0cfe.xml
|       |   |   |-- ABC00443.jpg
|       |   |   |-- ABC00443.xml
|       |   |   |-- hydrocodone.004060367.large.jpg
|       |   |   |-- hydrocodone.004060367.large.xml
|       |   |   |-- MDQ08030.jpg
|       |   |   |-- MDQ08030.xml
|       |   |   |-- MJR19820.jpg
|       |   |   |-- MJR19820.xml
|       |   |   |-- null_1447885738791791-2.jpg
|       |   |   |-- null_1447885738791791-2.xml
|       |   |   |-- null_1484795219615615.jpg
|       |   |   |-- null_1484795219615615.xml
|       |   |   |-- null_1564366361939939-2.png
|       |   |   |-- null_1564366361939939-2.xml
|       |   |   |-- null_15815607080077.png
|       |   |   |-- null_15815607080077.xml
|       |   |   |-- PGO04030.jpg
|       |   |   |-- PGO04030.xml
|       |   |   |-- WAL01040.jpg
|       |   |   |-- WAL01040.xml
|       |   |   |-- WAL02920.jpg
|       |   |   |-- WAL02920.xml
|       |   |   |-- WHR01500.jpg
|       |   |   |-- WHR01500.xml
|       |   |   |-- WRH01500.jpg
|       |   |   `-- WRH01500.xml
|       |   |-- 1591599494251_833035.jpg
|       |   |-- 1592140107503_806057.jpg
|       |   |-- 1598292221212_49725.jpg
|       |   |-- 24426138944_7073fe0541_o.jpg
|       |   |-- 766x415_Naproxen_and_Acetaminophen_Use-1-732x415.jpg
|       |   |-- GettyImages-1148848406edit-732x549.jpg
|       |   `-- WAL01751.jpg
|       |-- models
|       |   `-- my_ssd_mobilenet_v2_fpnlite
|       |       |-- eval
|       |       |   |-- events.out.tfevents.1603176318.AIDevelopment.6710.1396.v2
|       |       |   |-- events.out.tfevents.1603177687.AIDevelopment.10465.1396.v2
|       |       |   `-- events.out.tfevents.1603178132.AIDevelopment.12246.1396.v2
|       |       |-- train
|       |       |   |-- events.out.tfevents.1603116735.AIDevelopment.18634.1482.v2
|       |       |   |-- events.out.tfevents.1603174036.AIDevelopment.2665.1482.v2
|       |       |   `-- events.out.tfevents.1603268188.AIDevelopment.13897.1482.v2
|       |       |-- checkpoint
|       |       |-- ckpt-20.data-00000-of-00001
|       |       |-- ckpt-20.index
|       |       |-- ckpt-21.data-00000-of-00001
|       |       |-- ckpt-21.index
|       |       |-- ckpt-22.data-00000-of-00001
|       |       |-- ckpt-22.index
|       |       |-- ckpt-23.data-00000-of-00001
|       |       |-- ckpt-23.index
|       |       |-- ckpt-24.data-00000-of-00001
|       |       |-- ckpt-24.index
|       |       |-- ckpt-25.data-00000-of-00001
|       |       |-- ckpt-25.index
|       |       |-- ckpt-26.data-00000-of-00001
|       |       |-- ckpt-26.index
|       |       `-- pipeline.config
|       |-- pre-trained-models
|       |   `-- ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8
|       |       |-- checkpoint
|       |       |   |-- checkpoint
|       |       |   |-- ckpt-0.data-00000-of-00001
|       |       |   `-- ckpt-0.index
|       |       |-- saved_model
|       |       |   |-- variables
|       |       |   |   |-- variables.data-00000-of-00001
|       |       |   |   `-- variables.index
|       |       |   `-- saved_model.pb
|       |       `-- pipeline.config
|       |-- 160714_Pills_2_1080p.mp4
|       |-- 160714_Pills_4_1080p.mp4
|       |-- 160714_Pills_6_1080p.mp4
|       |-- exporter_main_v2.py
|       |-- model_main_tf2.py
|       |-- TF-image-object-counting.py
|       |-- TF-image-od.py
|       |-- TF-video-object-counting.py
|       |-- TF-video-od.py
|       `-- TF-webcam-opencv.py
`-- Training-a-Custom-TensorFlow-2.X-Object-Detector-master.zip
```
