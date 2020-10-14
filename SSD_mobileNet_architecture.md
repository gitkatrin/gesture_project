# Mobile Net Architecture

1) relationship inside one channel <math> D<sub>K</sub> * D<sub>K</sub>* M <math>
    - first filter on first channel, second filter on second channel, ...
  
2) relationship between channels <math> D<sub>K</sub><sup>2</sup> * M * D<sub>F</sub><sup>2</sup> * N <math> -> with N filters
    - stack layer into a block
    
        <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/regular_conv.png" width="180">
  
3) b) in image below: <math> D<sub>K</sub><sup>2</sup> * D<sub>F</sub><sup>2</sup> * M <math>     c) in image below: <math> M * D<sub>K</sub><sup>2</sup> * N<math>

      <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/mobile_net.png" width="220">

# SSD Architecture (Single Shot Detection; [SSD Paper Link](https://arxiv.org/pdf/1512.02325.pdf "SSD Paper"))

   <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/SSD_Model.jpeg" width="800">
   
- 7 Layer, each layer has a different size of feature map (e.g. Layer one has a feature map size of 38x38)

1.) Detection of 4 or 6 default boxes for each cell in every feature map -> summarized: 8732 Detections per Class

2.) Hard Negative Mining and Non-Maximum Suppression
- **while training**: give model the information of the ground trouth and do Hard Negative Mining

   <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/Hard%20Negative%20Mining.png" width="500">

- **at inference time**: Non-Maximum Suppression (NMS)

   <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/Non-Maximum%20Suppression%20.jpeg" width="200">
