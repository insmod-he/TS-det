#include <algorithm>
#include <cfloat>
#include <vector>
#include <map>

#include "string.h"
#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iostream>
#include "caffe/util/benchmark.hpp"

#ifdef __GNUC__
#include <ext/hash_map>
#else
#include <hash_map>
#endif

//#define HLHE_DEBUG

namespace std
{
using namespace __gnu_cxx;
}


namespace caffe {

extern int g_HardMiningStatus;
extern int g_NormalStatus;
extern int g_mining_status;

template <typename Dtype>
void SoftmaxWithLossOLHMLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);

  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  // setup a softmax layer -hlhe

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_)
  {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
  noisy_flip_= this->layer_param_.loss_param().noisy_flip();

  // Init T_
  if (noisy_flip_){
    T01_ = 0.1;
    T10_ = 0.1;
    T00_ = 1.0 - T01_;
    T11_ = 1.0 - T10_;
    CHECK_EQ((T00_+T01_), 1.0);
    CHECK_EQ((T10_+T11_), 1.0);
    CHECK_GE(T00_, 0.0);
    CHECK_GE(T01_, 0.0);
    CHECK_GE(T10_, 0.0);
    CHECK_GE(T11_, 0.0);
    CHECK_EQ(bottom[0]->shape()[1], 2);
  }
  else{
      for( int k=0; k<5; k++){
        LOG(INFO)<<"Attension!!!!!\n";
        sleep(1);
      }
      LOG(INFO)<<"Use normal softmax loss!\n";
      sleep(5);
  }
}

template <typename Dtype>
void SoftmaxWithLossOLHMLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  // outer_num = batch_sizen
  outer_num_ = bottom[0]->count(0, softmax_axis_);

  // inner_num = HxW
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
//   LOG(INFO)<<"outer_num_:"<<outer_num_;
//   LOG(INFO)<<"inner_num_:"<<inner_num_;

  if (top.size() >= 2)
  {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossOLHMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();  // softmax_layer -> prob_ hlhe
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;  // CxHxW
  int count = 0;
  Dtype loss = 0;

  // outer_num -> batch_size
  for (int i = 0; i < outer_num_; ++i)
  {
    // inner_num -> HxW
    for (int j = 0; j < inner_num_; j++)
    {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_)
      {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));

      // negative maximum likelihood
      if (noisy_flip_){
          const Dtype* bottom0_data = bottom[0]->cpu_data();
          int p0_idx = i * dim + 0*inner_num_ + j;
          int p1_idx = i * dim + 1*inner_num_ + j;
          float b0 = bottom0_data[p0_idx];
          float b1 = bottom0_data[p1_idx];
          float maxb = std::max(b0,b1);
          float eb0 = exp(b0-maxb);
          float eb1 = exp(b1-maxb);
          //float my_prob0 = eb0/(eb0+eb1);
          //std::cout<<"my_prob0: "<<my_prob0<<"\tprob0: "<<prob_data[p0_idx]<<"\n";

          if( 0==label_value ){
            loss -= log(std::max( (T00_*eb0+T10_*eb1) / (eb0+eb1), float(FLT_MIN)));
          }
          else if( 1==label_value ){
            loss -= log(std::max( (T11_*eb1+T01_*eb0) / (eb0+eb1), float(FLT_MIN)));
          }
      }
      else{
          loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      }
      ++count;
    }
  }
  if (normalize_)
  {
    top[0]->mutable_cpu_data()[0] = loss / count;
  }
  else
  {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }

  // output probability
  if (top.size() == 2)
  {
    top[1]->ShareData(prob_);
  }
}


//======================================================================
void fprint_val(FILE* fd, float val)
{
    assert(fd);
    fprintf(fd, "%f ", val);
}
void fprint_val(FILE* fd, double val)
{
    assert(fd);
    fprintf(fd, "%f ", val);
}

void fprint_val(FILE* fd, int val)
{
    assert(fd);
    fprintf(fd, "%d ", val);
}

template<typename T>
void fprintf_T(FILE* fd, T val)
{
    fprint_val(fd, val);
}


template <typename Dtype>
int print_blob_txt(Blob<Dtype>& b, int batch_idx,
                   int ch_idx,const char* txt_name, bool is_diff=false)
{
    using namespace std;
    assert(txt_name);
    assert(b.shape().size()==4);

    FILE* pfd = fopen(txt_name, "w");
    int ch= b.shape()[1];
    int h = b.shape()[2];
    int w = b.shape()[3];
    int dim = ch*h*w;
    int pixel_num = h*w;

    Dtype  val = 0;
    const Dtype* pdata = NULL;

    if(is_diff)
    {
        pdata = b.cpu_diff();
    }
    else
    {
        pdata = b.cpu_data();
    }

    assert(pdata);
    for (int i=0; i<h; i++)
    {
        for (int j=0; j<w; j++)
        {
            val = pdata[batch_idx*dim + ch_idx*pixel_num + i*w + j];
            fprintf_T(pfd, val);
        }
        fprintf(pfd, "\n");
    }

    fclose(pfd);

    return 0;
}

// add hlhe
int my_rand(int n)
{
    return rand()%n;
}


template <typename Dtype>
void SoftmaxWithLossOLHMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
  static unsigned long tic = 0;
  if (propagate_down[1])
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0])
  {
    HardMiner minner;
    minner.set_status(g_mining_status);

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);

    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;

    // bottom[0] is not feature
    int class_num = bottom[0]->shape()[1];
    vector<const Dtype*> lab_vec;
    lab_vec.push_back(label);

    minner.find_negatives(prob_, lab_vec, outer_num_, inner_num_,class_num);
    minner.set_ignores();
    //minner.print_ignore();

    int count = 0;
    const Dtype* dist = bottom[2]->cpu_data();
    for (int i = 0; i < outer_num_; ++i)
    {
      //#define HLHE_DEBUG
      #ifdef HLHE_DEBUG
      int pic_pos_cnt = 0;
      int pic_neg_cnt = 0;
      #endif

      cv::Mat show_mat1(120, 160, CV_8U, cv::Scalar(255));
      for (int j = 0; j < inner_num_; ++j)
      {
        int label_value = static_cast<int>(label[i * inner_num_ + j]);
   
        // set gradient to 0
        if ((has_ignore_label_ && label_value == ignore_label_) ||
             minner.ignore_or_not(i,j, label_value))
        {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c)
          {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
        else
        {
          if( noisy_flip_ ){
              int batch_start = i*2*inner_num_;
              int p0_idx = batch_start + 0 * inner_num_ + j;
              int p1_idx = batch_start + 1 * inner_num_ + j;
              const float max_inside_dis = 8.0;
              const float max_outside_dis = 12.0;

              CHECK_EQ(bottom.size(), 3);
              CHECK_EQ(bottom[2]->shape()[1], 2);
              float inside_dis  = dist[p0_idx];  // inside distance
              float outside_dis = dist[p1_idx];  // outside distance
              
              // calc flip probability
              if( inside_dis<max_inside_dis && outside_dis<max_outside_dis \
                    && inside_dis>4 && outside_dis>4 ){
                label_value = ignore_label_;
              }

              if( outside_dis<max_outside_dis && outside_dis>inside_dis ){
                  float r = outside_dis/max_outside_dis*0.5;
                  T00_ = 0.5 + r;
                  T11_ = 0.5 - r;
              }
              else if( inside_dis<max_inside_dis && inside_dis>outside_dis ){
                  float r = inside_dis/max_inside_dis*0.5;
                  T00_ = 0.5 - r;
                  T11_ = 0.5 + r;
              }
              else{
                  T00_ = 1.0;
                  T11_ = 1.0;
              }
              T01_ = 1.0 - T00_;
              T10_ = 1.0 - T11_;

              // calc gradicent
              const Dtype* bottom0_data = bottom[0]->cpu_data();
              float eb0 = exp(bottom0_data[p0_idx]);
              float eb1 = exp(bottom0_data[p1_idx]);
              float prob0 = prob_data[p0_idx];
              float prob1 = prob_data[p1_idx];
              float grad_b0 = 0.0, grad_b1 = 0.0;

              if( 0==label_value ){
                  // hua jian! c0 = (T10_-T00_) / (T00_*eb0+T10_*eb1)
                  float v = (T10_-T00_) / (T00_*eb0+T10_*eb1);
                  grad_b0 = prob0 * eb1 * v;
                  grad_b1 = prob1 * eb0 * v * -1.0;
              }
              else if( 1==label_value ){
                  float v = (T11_-T01_) / (T11_*eb1+T01_*eb0);
                  grad_b0 = prob0 * eb1 * v;
                  grad_b1 = prob1 * eb0 * v * -1.0;
              }
              bottom_diff[p0_idx] = grad_b0;
              bottom_diff[p1_idx] = grad_b1;
          }
          else{
              bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          }
          ++count;

        //==========================================
        #ifdef HLHE_DEBUG
                if(label_value==0)
                {
                    pic_neg_cnt+=1;
                }
                else if(label_value==1)
                {
                    pic_pos_cnt+=1;
                }
                else
                {
                    assert(1);
                }
        #endif
        //==========================================
        }
      }

      if( tic%1000==0 ){
        for( int y=0; y<120; y++){
            for( int x=0; x<160; x++){
                int idx = y*160 + x;
                int v = minner.ignore_or_not(i,idx, 0);
                show_mat1.at<unsigned char>(y,x) = v*255;
            }
        }
        string save_root = "/data2/HongliangHe/work2017/TrafficSign/node113/noisy_hm/my/0515_test/debug_imgs/";
        string save_name = ""; string rnd_name = "";
        char ss[256] = {}; sprintf(ss, "%d", tic); rnd_name = ss;
        save_name = save_root + rnd_name + "_in.jpg";
        imwrite(save_name.c_str(), show_mat1);
        LOG(INFO)<<save_name<<" saved!";
      }

#ifdef HLHE_DEBUG
      LOG(INFO)<<"pic_neg_cnt:"<<pic_neg_cnt;
      LOG(INFO)<<"pic_pos_cnt:"<<pic_pos_cnt;
#endif
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_)
    {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    }
    else
    {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
  tic += 1;
}


//#ifdef CPU_ONLY
//STUB_GPU(SoftmaxWithLossOLHMLayer);
//#endif


HardMiner::HardMiner()
{
    pIgnore_falgs_ = NULL;
}

HardMiner::~HardMiner()
{
    if(pIgnore_falgs_)
    {
        delete[] pIgnore_falgs_;
        pIgnore_falgs_ = NULL;
    }
}

template <typename Dtype>
int HardMiner::find_negatives(Blob<Dtype>& prob,
                              vector<const Dtype*>& lab_v,
                              int batch_num,
                              int inner_num,
                              int class_num)
{
    assert(class_num==2);
    const Dtype* prob_dat = prob.cpu_data();
    for(int batch_idx=0; batch_idx<batch_num; batch_idx++)
    {
        int gray_zone_num = 0;
        int pos_num = 0;
        int neg_num = 0;

        map<float, int > neg_samples;
        for( int k=0; k<inner_num; k++)
        {
            int lab = static_cast<int>(lab_v[0][batch_idx*inner_num + k]);

            // negative samples
            if(lab==0)
            {
                float neg_prob = prob_dat[batch_idx*inner_num*class_num + k];
                neg_prob += rand()*1.0/INT_MAX*1e-4;
                neg_samples[neg_prob] = k;
                neg_num += 1;
            }
            else if(lab==1)
            {
                pos_num += 1;
            }
            else if(lab==2)
            {
                gray_zone_num+=1;
            }
            else
            {
                LOG(INFO)<<"Abnormal label:"<<lab;
                assert(1);
            }
        }
        pos_num_vec_.push_back(pos_num);
        negatives_vec_.push_back(neg_samples);
    }
    batch_size_ = batch_num;
    inner_size_ = inner_num;
    channel_    = class_num;

    return 0;
}

// ignore non-important samples
int HardMiner::set_ignores()
{
    assert(pIgnore_falgs_==NULL);
    const int MIN_POS_NUM = 200;
    pIgnore_falgs_ = new char[batch_size_*inner_size_];
    assert(pIgnore_falgs_);
    caffe_memset(batch_size_*inner_size_*sizeof(char), 1,pIgnore_falgs_);
    //print_ignore();

    static int cnt = 0;
    if (cnt%100==0)
    {
        if (status_==g_NormalStatus)
        {
            LOG(INFO)<<"Normal";
        }
        else
        {
            LOG(INFO)<<"HardMining";
        }
    }
    cnt += 1;

    for(int batch_idx=0; batch_idx<batch_size_; batch_idx++)
    {
        bool has_no_positive = false;
        int pos_num = pos_num_vec_[batch_idx];  // now 1:2
        if(pos_num==0)
        {
            has_no_positive = true;
            pos_num = MIN_POS_NUM;
        }

        int idx = 0;
        vector<int> non_hard_vec;
        std::map<float, int >::iterator it = negatives_vec_[batch_idx].begin();
        for(; it!=negatives_vec_[batch_idx].end(); ++it)
        {
            int sample_idx = it->second;
            sample_idx += batch_idx*inner_size_;

            if(idx<pos_num){
                pIgnore_falgs_[sample_idx] = 0;
            }
            else{
                non_hard_vec.push_back(sample_idx);
            }
            idx +=1;
        }

        // half non-hard negative samples
        if( false==has_no_positive ){
            std::random_shuffle(non_hard_vec.begin(), non_hard_vec.end());
            int normal_neg_num = std::min(pos_num, int(non_hard_vec.size()));
            for( int k=0; k<normal_neg_num; k++){
                pIgnore_falgs_[ non_hard_vec[k] ] = 0;
            }
        }
    }

    return 0;
}

bool HardMiner::ignore_or_not(int batch_idx, int pos, int lab_val)
{
    if(lab_val==1)
        return false;

    assert(pIgnore_falgs_);
    assert(batch_idx>=0 && batch_idx<batch_size_);
    assert(pos>=0 && pos<inner_size_);

    char v = pIgnore_falgs_[batch_idx*inner_size_ + pos];
    return v;
}

int HardMiner::print_ignore()
{
    if(pIgnore_falgs_)
    {
        for(int k=0; k<batch_size_; k++)
            for(int i=0; i<inner_size_; i++)
            {
                LOG(INFO)<<(int)pIgnore_falgs_[k*inner_size_+i]<<"\n";
            }
        LOG(INFO)<<"\n";
    }
    return 0;
}

int HardMiner::set_status(int s)
{
    status_ = s;
    return 0;
}


INSTANTIATE_CLASS(SoftmaxWithLossOLHMLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLossOLHM);

}  // namespace caffe
