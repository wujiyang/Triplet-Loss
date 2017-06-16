#ifndef CAFFE_RANK_HARD_LOSS_LAYER_HPP_
#define CAFFE_RANK_HARD_LOSS_LAYER_HPP_

#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    
  template <typename Dtype>
  class RankHardLossLayer : public LossLayer<Dtype> {
  public:
    explicit RankHardLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "RankHardLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
  
    static int MyRandom(int i);
    void set_mask(const vector<Blob<Dtype>*>& bottom);
  
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
                              
    Blob<Dtype> diff_ ;
    Blob<Dtype> dis_ ;
    Blob<Dtype> mask_ ;
  };

}

#endif  //CAFFE_RANK_HARD_LOSS_LAYER_HPP_