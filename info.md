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
  - Key challange: number of objects is unknown (How many bounding boxes? How many classifictions?)
    - proviode a fixed number of bounding boxes+classifications(maximum)
    - classify bounding boxes as "object" or "not an object" -> only considering "objects" to produce variable number of boxes+classifications
  - Training:
    - True classifications and bounding boxes are known
    - Secify the output:
      - Split imput data into 3x3 cells
        - for each cell:
         
            <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/Training_vector.PNG" width="250">
         
        - Cell that contains the center point of the object is associated to the object:
          -> identify cell with center point of bounding box
          -> conpare bounding box with different anchor boxes in this cell
          -> associate object to ancher box with most sililar shape (higherst IoU)
        - Multiple anchor boxes in each cell (anchor box: initial guess for a bounding box with fixed size)
          -> output one y for every anchor box
  - Testing:
    - basic probelm: obtain too many bounding boxes; trainined network would output a vector y for every anchor box
    - remove all bounding boxes which are p < 0,5 (p: Probability that its an object)
      - normally left with a few bounding boxes for each object (many anchor boxes overlap with object because of the 19x19 grid)
    - Non-maximum suppression:
      1. Remove all anchor boxes: p < 0,5
      2. Find anchor box with the largest probability of being an object and store that to set of object detections (bounding box i)
      3. Remove all anchor boxes:
          - the same most probable class is the same than bpunding box i
          - bounding box overlaps substantially with bounding box i 
      4. Repeat until fixed bounding boxes
  - Paper: 
    - The convolutional model for predicting detections is different for each feature layer
    - Default boxes and aspect ratios:
      - associate set of default bounding boxes with each feature map cell (one cell with set of bounding boxes) for multiple feature maps at the top of the network
      - default boxes tile the feature map in a convolutional manner -> position of each box is fixed to corresponding cell
      - in each feature map cell: 
        - predict **offsets relative** to the default box shapes
        - per-class **scores** that indicate the presence of a class instance (in each box)
      - compute relative to original default box shape, for each box out of k at a given location the c class scores and 4 offsets
      
            -> (c + 4)k filters, applied around each location in the feature map -> (c + 4)kmn outputs (for m x n feature map)
      - like anchor boxes (Faster R-CNN) but apply them on several feature maps of different resolutions
      - allowing different default box shapes in severat feature maps -> discretize the space of possible output box shapes
