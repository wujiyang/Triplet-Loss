## Triplet Loss for FaceRecognition

### Introduction  

Modified from [xiaolonw/caffe-video_triplet](https://github.com/xiaolonw/caffe-video_triplet), update the source code to  
fit the new verison of [BVLC/caffe](https://github.com/BVLC/caffe).   

Details:   
1. add ##pair_size## parameter for image_data_layer to shuffle training example in pair. Normally, pair_size = 2.  
in the prototxt:  
```
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "label"
  include { 
    phase: TRAIN
  }
  transform_param {
    scale: 0.0078125
    mirror: true
  }
  image_data_param {
    source: "/media/wujiyang/data/FaceData/train.txt"
    batch_size: 128
    shuffle: true
    pair_size: 2
  }
}   
```  
2. add ##.hpp## file for norm_layer and rank_hard_loss layer   

### Usage
In the prototxt  
```
layer {
  name: "triplet-loss"
  type: "RankHardLoss" 
  rank_hard_loss_param{
    neg_num: 4
    pair_size: 2
    hard_ratio: 0
    rand_ratio: 1.0
    margin: 0.1
  }
  bottom: "norml2"
  bottom: "label"
} 
```  
when training model with the CASIA-WebFace， trying to  set margin = 0.1 or 0.2.   
In my experiments, when margin = 1,the loss didn't converge at all. 