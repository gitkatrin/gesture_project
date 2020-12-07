#!/usr/bin/env python
# coding: utf-8
"""
Object Detection (On Image) From TF2 Saved Model
=====================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_mobilenet_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
parser.add_argument('--image', help='Name of the single image to perform detection on',
                    default='./img')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.3)
                    
args = parser.parse_args()
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = args.image


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = args.model

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = args.labels

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(args.threshold)

# LOAD THE MODEL ==============================================================

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# LOAD IMAGE ==================================================================

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
from collections import namedtuple
import os



def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """    
    
    return np.array(Image.open(path))

print('Running inference for {}... '.format(IMAGE_PATHS), '\n')

# ORDNER EINLESEN & BILDER ABFRAGEN ===========================================

import glob

def load_img_folder(img_folder_path):
    
    img_names = []
    img_data = []
    
    for img  in glob.glob(img_folder_path):
        
        # create image name list
        img_name = img.split('/')[-1]
        img_names.append(img_name)
        
        #create image data list
        img_data.append(cv2.imread(img))
        
    # create image name array
    file_names = np.genfromtxt(img_names ,delimiter=',', usecols=0, dtype=str)
    
    return img_names, img_data, file_names
    
# CSV DATEI EINLESEN & BILDNAMEN VERGLEICHEN ==================================

def read_csv(image_name): # ein Bild kommt hier rein
    
    csv_data = np.genfromtxt('test_labels_2.csv',delimiter=',',skip_header=1, 
                             usecols= (4, 5, 6, 7), missing_values = {0: str})
    csv_names = np.genfromtxt('test_labels_2.csv',delimiter=',', skip_header=1, 
                              usecols=0, dtype=str)
    
    truth_boxes = []

    for i in range(len(csv_names)):   # lenght = 4320
        
        if image_name in csv_names[i]:
            truth_boxes.append(csv_data[i])

    return truth_boxes

# CALCULATE IOU ===============================================================


def batch_iou(boxA, boxB):
    
    # COORDINATES OF THE INTERSECTION BOXES
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
   
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou


# WRITE NEW CSV FILE WITH CALCULATED IOU ======================================

import csv
        
def create_new_csv(data): 
       
    with open('output.csv', 'w') as csv_file:
        
        # Überschriften erstellen
        fieldnames = ['image name', 'gt_xmin', 'gt_ymin', 'gt_xmax', 'gt_ymax', 
                      'pred_xmin', 'pred_ymin', 'pred_max', 'pred_ymax', 'iou', 'score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(data)


#==============================================================================
k = 0

img_names, img_data, file_names = load_img_folder('./img/*.jpg')

data_for_csv = []


while k < len(img_names):
    
    # truth_boxes = [array([557., 505., 817., 719.]), array([ 937.,  385., 1130.,  555.]), array([760., 371., 931., 527.])]
    truth_boxes = read_csv(str(img_names[k]))
    image = img_data[k]
   
    
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    imH, imW, _ = image.shape
    #image_expanded = np.expand_dims(image_rgb, axis=0)
    
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    
    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    count = 0
    

    for i in range(len(scores)):
        if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
            #increase count
            count += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
            # Draw label
            object_name = category_index[int(classes[i])]['name'] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            pred_box = np.array([xmin, ymin, xmax, ymax])
            score = scores[i]*100
            
        
            for box in truth_boxes:
                
                iou = batch_iou(box, pred_box)
                
                if (iou > 0.5):
                    real_truth_box = box
                    minx = int(real_truth_box[0])
                    miny = int(real_truth_box[1])
                    maxx = int(real_truth_box[2])
                    maxy = int(real_truth_box[3])
                    cv2.rectangle(image, (minx,miny), (maxx,maxy), (255, 0, 0), 2)
                
            iou = batch_iou(real_truth_box, pred_box)       
            
            print('Image:', img_names[k])                
            print('Predictionbox:', pred_box, ' ', 'Ground truth box:', real_truth_box, '\033[1m' +'\nIoU=', iou, '\033[0m')
           
            
            data = [str(img_names[k]), minx, miny, maxx, maxy, xmin, ymin, xmax, ymax, iou, score]
            data_for_csv.append(data)
            # [['CARDS_LIVINGROOM_H_S_frame_2149.jpg', '[728.0, 470.0, 1008.0, 719.0]', '[ 738  476 1016  717]', '0.9085141163106419'], 
            # ['CARDS_LIVINGROOM_H_S_frame_2149.jpg', '[674.0, 400.0, 847.0, 510.0]', '[674 403 874 501]', '0.7834629553827261'], ...
            #print(data_for_csv)
            
            
        cv2.putText(image, "IoU: {:.4f}".format(iou), (xmax, label_ymin-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    k += 1
          

    cv2.putText (image,'Total Detections : ' + str(count),(10,25),cv2.FONT_HERSHEY_SIMPLEX,1,(70,235,52),2,cv2.LINE_AA)
    create_new_csv(data_for_csv)
    print('\nDone')
    
    # display output image and destroy after 5 seconds
    cv2.imshow('Object Counter', image)
    cv2.waitKey(0) 
    cv2.destroyWindow('Object Counter')

    
       
# CLOSES WINDOW ONCE KEY IS PRESSED
cv2.waitKey(0)
# CLEANUP
cv2.destroyAllWindows()

