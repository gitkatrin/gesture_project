# Mobile Net Architecture

1) relationship inside one channel <math> D<sub>K</sub> * D<sub>K</sub>* M <math>
    - first filter on first channel, second filter on second channel, ...
  
2) relationship between channels <math> D<sub>K</sub><sup>2</sup> * M * D<sub>F</sub><sup>2</sup> * N <math> -> with N filters
    - stack layer into a block
    
        <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/regular_conv.png" width="180">
  
3) b) auf Abbildung: <math> D<sub>K</sub><sup>2</sup> * D<sub>F</sub><sup>2</sup> * M <math>     c) auf Abbildung: <math> M * D<sub>K</sub><sup>2</sup> * N<math>

        <img src="https://github.com/gitkatrin/gesture_project/blob/master/images/mobile_net.png" width="220">

