**Artificial Intelligence Frameworks:**
-  Frameworks:
  - Caffee
  - Theano
  - PyTroch
  - Distbelief
- Tensorflow: Set of APIs and data structures


**Website for datasets:**
- seach for datasets:
  - https://www.kaggle.com/datasets
  - https://cocodataset.org/#explore
  - http://www.image-net.org/
  - https://archive.ics.uci.edu/ml/datasets.php
  - http://yann.lecun.com/exdb/mnist/
- search for websites: 
  - https://datasetsearch.research.google.com/
  - https://www.kdnuggets.com/datasets/index.html

**Pre-Trained Networks in Keras:**
- VGG16
- VGG19
- Resnet50
- Inception V3
- Xception
- MobileNet

**Object Detection Approaches:**
- RCNN (https://arxiv.org/abs/1311.2524)
- Fast RCNN (https://arxiv.org/abs/1504.08083)
- Faster RCNN (https://arxiv.org/abs/1506.01497)
- Yolo - you only look once
- Implementation unsing TensorFlow
- SSD (https://arxiv.org/pdf/1512.02325.pdf):
  - Key challange: nujmber of objects is unknown (How many bounding boxes? How many classifictions?)
    - proviode a fixed number of bounding boxes+classifications(maximum)
    - classify bounding boxes as "object" or "not an object" -> only considering "objects" to produce variable number of boxes+classifications
  - Training:
    - Secify the output:
      - Split imput data into 3x3 cells
        - for each cell:
          ![Alt-Text](C:/Users/katrin.gloewing/Desktop/092020-112020_Praktikum/Training_vector)
    
    
