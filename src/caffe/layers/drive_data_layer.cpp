#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdint.h>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include <algorithm>
#include <math.h>
#include <strstream>

using namespace std;
namespace caffe {

const float INF_DIS = 1e6;
const float IGNORE_DIS = 1e5;

const int kNumData = 1;
const int kNumLabels = 1;
const int kNumBBRegressionCoords = 4;
const int kNumRegressionMasks = 8;
const int Lab4ClsChanelIdx    = kNumRegressionMasks;   // Ϊ�����ֱ߽����ع���mask/label
const int NewLabelChannelNum  = Lab4ClsChanelIdx+1;

const int PATH1_RECIPTION_FILD = 114;  // path1 reception file hlhe
const int PATH2_RECIPTION_FILD = 210;
const int PATH1_MAX_OBJ_SIZE   = 72;
const int PATH2_MIN_OBJ_SIZE   = 64;
const int PATH2_MAX_OBJ_SIZE   = 110;
const int PATH3_MIN_OBJ_SIZE   = 100;
const float GRAY_ZONE_LAB      = 2.0;
const float IGNORE_EDGE_LAB    = 3.0; // ���Ա��Ǳ�־��ΧӦ�ú��Ե�����
const float TRAFFIC_SIGN_LAB   = 1.0;
const float BACKGROUND_LAB     = 0.0;
const int MASK_NUM = 3;
const float INVALIDE_BBOX_IDX = -1.0;

// ��const����֮��,��Ȼ�Ͳ��ܵ���ȫ�ֱ�����~
int g_HardMiningStatus   = 0;
int g_NormalStatus       = 1;
int g_mining_status = g_NormalStatus;

template <typename Dtype>
DriveDataLayer<Dtype>::DriveDataLayer(const LayerParameter& param)
  : DataLayer<Dtype>(param)
{
}

template <typename Dtype>
void DriveDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  int hm_perid = this->layer_param().drive_data_param().hm_perid();
  float ign_bb_scale = this->layer_param().drive_data_param().ignore_bbox_scale();
  lab_auxi_.set_perid(hm_perid);
  lab_auxi_.set_bbox_scale(ign_bb_scale);
  srand(time(0));
  const int batch_size = this->layer_param_.data_param().batch_size();
  DriveDataParameter param = this->layer_param().drive_data_param();
  // Read a data point, and use it to initialize the top blob.
  LOG(INFO)<<"top.size() "<<top.size();

  vector<int> top_shape(4,0);
  int shape[4] = {batch_size,
                  3,
                  (int)param.cropped_height(),
                  (int)param.cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));

  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
  {
    this->prefetch_[i].data().Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  if (param.genlist_file() != "")
  {
      LOG(FATAL)<<"Bad parameter:"<<param.genlist_file();
  }
  // label
  if (this->output_labels_)
  {
    vector<int> label_shape(4,0);
    vector<int> type_shape(4,0);
    vector<int> mask_shape(4,0);
    int h = param.tiling_height() * param.label_resolution();
    int w = param.tiling_width() * param.label_resolution();

    int shape[4] = {
        batch_size,
        NewLabelChannelNum, //kNumRegressionMasks, modify0914
        h,                  // This is the dim after tiling -hlhe
        w };                // 20x8=160, 15x8=120

    int shape_mask[4] = {
        batch_size,
        MASK_NUM,           // mask1 mask2 mask3
        h,
        w };

    int shape_type[4] = {
        batch_size, 1,
        param.tiling_height() * param.catalog_resolution(),
        param.tiling_width() * param.catalog_resolution()
    };
    memcpy(&label_shape[0], shape, sizeof(shape));
    top[1]->Reshape(label_shape);

    memcpy(&type_shape[0], shape_type, sizeof(shape_type));
    memcpy(&mask_shape[0], shape_mask, sizeof(shape_mask));
    top[2]->Reshape(mask_shape);     // Because there has no type loss. -hlhe

    // add 0515, distance data
    vector<int> dis_shape;
    dis_shape.push_back(batch_size);
    dis_shape.push_back(2);          // inside-outside
    dis_shape.push_back(h);
    dis_shape.push_back(w);
    top[3]->Reshape(dis_shape);

    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
      this->prefetch_[i].label(0).Reshape(label_shape);
      this->prefetch_[i].label(1).Reshape(mask_shape);
      this->prefetch_[i].label(2).Reshape(dis_shape);
      this->prefetch_[i].label(3).Reshape(type_shape);
    }
    lab_auxi_.reset_gray_zone(h, w);
    //pfun_is_normal = &(lab_auxi_.is_normal_status());
  }

  static bool first_in_flag = true;
  if(first_in_flag)
  {
      first_in_flag = false;
      LOG(INFO)<<"resize_max:"<<param.resize_max()<<"  resize_min: "<<param.resize_min();
      LOG(INFO)<<"reco_min:"<<param.reco_min()<<"  reco_max:"<<param.reco_max();
  }
}

int Rand()
{
    return rand();
}

float rand_float()
{
    return rand()*1.0f / RAND_MAX;
}

bool check_cross(float l1, float r1, float l2, float r2)
{
    if (r1 < l2 || l1 > r2)
        return false;
    return true;
}



template<typename Dtype>
void mask_random_ignore(float prob, cv::Mat& mask, Dtype igr_val)
{
    for ( int i=0; i<mask.rows; i++)
    {
        for( int j=0; j<mask.cols; j++)
        {
            bool ignore = rand_float()<prob;
            if(ignore)
                mask.at<Dtype>(i, j) = igr_val;
        }
    }
}



int calc_cross(int x1,int x2,int y1,int y2, int a1,int a2,int b1,int b2)
{
    int xmin = std::max<int>(x1,a1);
    int ymin = std::max<int>(y1,b1);
    int xmax = std::min<int>(x2,a2);
    int ymax = std::min<int>(y2,b2);

    int w = std::max<int>(0, xmax-xmin);
    int h = std::max<int>(0, ymax-ymin);
    return w*h;
}

void get_cross(int x1,int x2,int y1,int y2, int a1,int a2,int b1,int b2, vector<int>& ret)
{
    int xmin = std::max<int>(x1,a1);
    int ymin = std::max<int>(y1,b1);
    int xmax = std::min<int>(x2,a2);
    int ymax = std::min<int>(y2,b2);
    ret.push_back(xmin);
    ret.push_back(ymin);
    ret.push_back(xmax);
    ret.push_back(ymax);
    return ;    
}

int bb_pushback(vector<vector<int> >&v, int x1,int x2,int y1, int y2)
{
    vector<int> bb;
    bb.push_back(x1);
    bb.push_back(x2);
    bb.push_back(y1);
    bb.push_back(y2);
    v.push_back(bb);

    return 0;
}

int rescale_bb(int& x1, int& x2, int& y1, int& y2, int h, int w, float scale)
{
    float dw = (x2-x1-1) * (scale-1)/2.0;
    float dh = (y2-y1-1) * (scale-1)/2.0;
    dw = std::max<float>(dw, 2.0);
    dh = std::max<float>(dh, 2.0);

    int new_x1 = x1 - int(dw+0.5);
    int new_x2 = x2 + int(dw+0.5);
    int new_y1 = y1 - int(dh+0.5);
    int new_y2 = y2 + int(dh+0.5);

    if (new_x1<0)
        new_x1 = 0;
    if (new_x2>=w)
        new_x2 = w-1;
    if (new_y1<0)
        new_y1 = 0;
    if (new_y2>=h)
        new_y2 = h-1;

    x1 = new_x1; x2 = new_x2;
    y1 = new_y1; y2 = new_y2;

    return 0;
}

// ��ʴ��עģ��
int mask_erode(cv::Mat& mask, int k=3)
{
    cv::Mat m = mask>-1.0;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                      cv::Size(k, k),
                                      cv::Point(-1,-1));
    //my_show_mat("Before Erode", m);
    cv::erode(m, m, element);
    //my_show_mat("After Erode", m);

    for(int i=0; i<mask.rows; i++)
    {
        for(int j=0; j<mask.cols; j++)
        {
            float m_val = m.at<float>(i,j);
            if(m_val==0.0)
            {
                mask.at<float>(i,j) = -1;
            }
        }
    }

    return 0;
}

vector<int> get_scaled_box(float x1, float y1, float x2, float y2, 
                    float scaling, float shrink, int lab_w, int lab_h){
    float w = x2 - x1;
    float h = y2 - y1;
    CHECK_GT(w, 0);
    CHECK_GT(h, 0);
    //LOG(INFO)<<"w:"<<w<<"\th:"<<h<<"  lab_w:"<<lab_w<<" lab_h:"<<lab_h   ;

    float shrink_factor = (1.0-shrink) / 2.0;
    int gxmin = cvFloor((x1 + w * shrink_factor) * scaling);  // gmin must < 640 -hlhe
    int gxmax = cvCeil ((x2 - w * shrink_factor) * scaling);   // gmax must < 480 -hlhe
    int gymin = cvFloor((y1 + h * shrink_factor) * scaling);
    int gymax = cvCeil ((y2 - h * shrink_factor) * scaling);
    //LOG(INFO)<<"gxmin:"<<gxmin<<"  gxmax:"<<gxmax<<"  gymin:"<<gymin<<"  gymax:"<<gymax;
    if(gxmin<0){
        gxmin = 0;
    }
    if (gymin<0){
        gymin = 0;
    }
    if( gxmax>=lab_w ){
        gxmax = lab_w-1;
    }
    if( gymax>=lab_h ){
        gymax = lab_h-1;
    }
    //LOG(INFO)<<"gxmin:"<<gxmin<<"  gxmax:"<<gxmax<<"  gymin:"<<gymin<<"  gymax:"<<gymax;

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    if (gxmin >= lab_w)   // lab_w=160 -hlhe
    {
      gxmin = lab_w - 1;
    }
    if (gymin >= lab_h)
    {
      gymin = lab_h - 1;
    }
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, lab_w);
    CHECK_LE(gymax, lab_h);

    // gxmin,fxmax -> mask grid range -hlhe
    // deal with label critical conditions -hlhe
    if (gxmin == gxmax)
    {
      if (gxmax < lab_w - 1)
      {
        gxmax++;
      }
      else if (gxmin > 0)
      {
        gxmin--;
      }
    }

    if (gymin == gymax)
    {
      if (gymax < lab_h - 1)
      {
        gymax++;
      }
      else if (gymin > 0)
      {
        gymin--;
      }
    }
    CHECK_LT(gxmin, gxmax);
    CHECK_LT(gymin, gymax);
    if (gxmax == lab_w)
    {
      gxmax--;
    }
    if (gymax == lab_h)
    {
      gymax--;
    }

    vector<int> ret;
    ret.push_back(gxmin);
    ret.push_back(gymin);
    ret.push_back(gxmax);
    ret.push_back(gymax);
    return ret;
}

// Todo: remove overlaped regoin
void find_instance_boundary_pts(vector<vector<int> >& pts_vec, vector<int> ori_box,
    cv::Mat box_mask, cv::Mat polygon_mask, cv::Mat ellipse_mask, bool has_poly, bool has_ellipse){
    int dx[] = {-1,0,1,1,1,0,-1,-1};
    int dy[] = {-1,-1,-1,0,1,1,1,0};
    int w = box_mask.cols;
    int h = box_mask.rows;
    CHECK_EQ(ori_box.size(), 4);

    for( int y=ori_box[1]; y<=ori_box[3]; y++){
        for( int x=ori_box[0]; x<=ori_box[2]; x++){
            int v1 = int(box_mask.at<float>(y,x));
            int v2 = int(polygon_mask.at<float>(y,x));
            int v3 = int(ellipse_mask.at<float>(y,x));
            if( has_poly ){
                v1 = v1 * int(v2==v1);
            }
            if( has_ellipse ){
                v1 = v1 * int(v3==v1);
            }

            //cout<<v1<<" ";
            if( v1<=0 ){
                continue;
            }
            
            for( int k=0; k<8; k++){
                int near_x = x + dx[k];
                int near_y = y + dy[k];
                if (near_x<0 || near_x>w-1 || near_y<0 || near_y>h-1 ){      
                    // Out of image or near pixel has 0 value
                    // vector<int> pt;
                    // pt.push_back(x);
                    // pt.push_back(y);
                    // pts_vec.push_back(pt);
                    // break;
                }
                else{
                    v1 = int(box_mask.at<float>(near_y,near_x));
                    v2 = int(polygon_mask.at<float>(near_y,near_x));
                    v3 = int(ellipse_mask.at<float>(near_y,near_x));
                    if( has_poly ){
                        v1 = v1 * int(v2==v1);
                    }
                    if( has_ellipse ){
                        v1 = v1 * int(v3==v1);
                    }
                    if( v1<=0 ){
                        vector<int> pt;
                        pt.push_back(x);
                        pt.push_back(y);
                        pts_vec.push_back(pt);
                        break;
                    }
                }
            }
        }
    }
}

void calc_distance_to_boundary(vector<int> big_box, vector<int> small_box, float obj_size,int ins_idx, vector<vector<int> > boundary_pts, cv::Mat box_mask, 
       cv::Mat polygon_mask, cv::Mat ellipse_mask, cv::Mat& inside_dist, cv::Mat& outside_dist, bool has_poly, bool has_ellipse){
    if (boundary_pts.size()==0){
        LOG(INFO)<<"boundary_pts.size()==0";
        return;
    }
    if( obj_size<=4 ){
        return;
    }

    CHECK_EQ(big_box.size(), 4);
    CHECK_EQ(small_box.size(), 4);
    CHECK_EQ(box_mask.rows, polygon_mask.rows);
    CHECK_EQ(box_mask.cols, ellipse_mask.cols);
    CHECK_EQ(box_mask.rows, inside_dist.rows);
    CHECK_EQ(box_mask.cols, outside_dist.cols);
    const float invalid_neg_dis = -0.5;
    float fixed_max_dis = 20.0;
    float max_dis = obj_size/8.0;
    max_dis = std::min(max_dis, fixed_max_dis);

    for( int y=big_box[1]; y<big_box[3]; y++){
        for( int x=big_box[0]; x<big_box[2]; x++){

            int pt_min_dis2 = INF_DIS;
            int pre_min_absx = INF_DIS, pre_min_absy = INF_DIS;
            for( int k=0; k<boundary_pts.size(); k++){
                int _x = boundary_pts[k][0];
                int _y = boundary_pts[k][1];
                int dx = std::abs(_x-x);
                int dy = std::abs(_y-y);
                if( dx>pre_min_absx && dy>pre_min_absy){
                    continue;
                }
                
                int dis2 = dx*dx + dy*dy;
                if (dis2<pt_min_dis2){
                    pt_min_dis2 = dis2;
                    pre_min_absx = dx;
                    pre_min_absy = dy;
                }
            }

            float dis = INF_DIS;
            if(pt_min_dis2<INF_DIS){
                dis = sqrt(pt_min_dis2);
            }

            int v1 = int(box_mask.at<float>(y,x));
            int v2 = int(polygon_mask.at<float>(y,x));
            int v3 = int(ellipse_mask.at<float>(y,x));
            if( has_poly ){
                v1 = v1 * int(v2==v1);
            }
            if( has_ellipse ){
                v1 = v1 * int(v3==v1);
            }
            CHECK_GT(obj_size, 1);
            //LOG(INFO)<<"dis:"<<dis;

            // "1" based index
            if( v1>0){
                if (v1 == ins_idx){  // inside the object
                    float inside_v = inside_dist.at<float>(y,x);
                    if( inside_v==INF_DIS ){
                        inside_dist.at<float>(y,x) = dis;
                        //cout<<dis<<" ";
                    }
                    else if( inside_v!=IGNORE_DIS ){
                        inside_dist.at<float>(y,x) = IGNORE_DIS;
                    }
                    outside_dist.at<float>(y,x) = invalid_neg_dis;
                }
                else{
                    //LOG(INFO)<<"x:"<<x<<" y:"<<y<<" v1: "<<v1<<" idx:"<<ins_idx<<" overlap";
                    inside_dist.at<float>(y,x) = IGNORE_DIS;
                }
            }
            else{ // outside the object
                float outside_v = outside_dist.at<float>(y,x);
                if( outside_v==INF_DIS ){
                    outside_dist.at<float>(y,x) = dis;
                }
                else if( outside_v!=IGNORE_DIS ){
                    outside_dist.at<float>(y,x) = IGNORE_DIS;
                }
                inside_dist.at<float>(y,x) = invalid_neg_dis;
            }
        }   
    }
}

int draw_mask(const DrivingData& data, cv::Mat& poly_mask, cv::Mat& ellipse_mask, 
        int idx, int w_off, int h_off, float scaling, float resize)
{
    cv::Scalar idcolor(idx+1);
    caffe::CarBoundingBox box = data.car_boxes(idx);
    if (box.poly_mask_size()>2)
    {
        std::vector<cv::Point2i> pts;
        for (int j=0; j<box.poly_mask_size(); j++)
        {
            caffe::FixedPoint p = box.poly_mask(j);
            pts.push_back(cv::Point2i((int) ((p.x()*resize-w_off)*scaling),
                                    (int) ((p.y()*resize-h_off)*scaling)));
        }
        std::vector< std::vector<cv::Point2i> > ptss;
        ptss.push_back(pts);
        cv::fillPoly(poly_mask, ptss, idcolor);
    }
    if (box.ellipse_mask_size() > 2)
    {
        std::vector<cv::Point2i> pts;
        for (int j=0; j<box.ellipse_mask_size(); j++)
        {
            caffe::FixedPoint p = box.ellipse_mask(j);
            pts.push_back(cv::Point2i((int) ((p.x()*resize-w_off)*scaling),
                                    (int) ((p.y()*resize-h_off)*scaling)));
        }
        int tmp=0;
        while (pts.size()<5)
            pts.push_back(pts[tmp++]);

        cv::RotatedRect rbox = cv::fitEllipse(pts);
        cv::ellipse(ellipse_mask, rbox, idcolor, -1);
    }
    return 0;
}

template<typename Dtype>
bool ReadBoundingBoxLabelToDatum(
        const DrivingData& data,
        Datum* datum,                    // -> output
        const int h_off,
        const int w_off,
        const float resize,
        const DriveDataParameter& param,
        float* label_type,
        bool can_pass, 
        int bid,
        vector<cv::Mat> &genpic,
        vector<int> &genpic_type,
        vector<cv::Mat> &picgen,
        vector<Dtype*> &my_mask_ptr,
        vector<vector<int> > &bbox_pts,  // װ������ʣ�µ�bbox(label�����ֱ���)
        bool normal_status,              // Hard mining״̬
        vector<Dtype* > &lab_dis_vec)
{
  bool have_obj = false;
  const int grid_dim = param.label_resolution();
  const int width = param.tiling_width();
  const int height = param.tiling_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const float half_shrink_factor = (1-param.shrink_prob_factor()) / 2;
  const float unrecog_factor = param.unrecognize_factor();
  const float scaling = static_cast<float>(full_label_width) \
    / param.cropped_width();
  //const float resize = param.resize();

  // fast check 
  // Todo: ignore small boxes and unrecognize boxes
  for (int i = 0; i < data.car_boxes_size(); ++i)
  {
    // skip bboxes that not focus on -hlhe
    if (i != bid)
        continue;

    // new box coordinate after resize -hlhe
    float xmin = data.car_boxes(i).xmin()*resize;
    float ymin = data.car_boxes(i).ymin()*resize;
    float xmax = data.car_boxes(i).xmax()*resize;
    float ymax = data.car_boxes(i).ymax()*resize;
    float ow = xmax - xmin;
    float oh = ymax - ymin;

    xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
    xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
    ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
    ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    // drop boxes that unrecognize
    if (w*h < ow*oh*unrecog_factor)
        continue;

    // drop boxes that are too small
    if (w < 4 || h < 4)
    {
      continue;
    }
    if (std::max(w,h) < param.train_min() || std::max(w,h) > param.train_max())
        continue;
    have_obj = true;
  }
  if (can_pass && !have_obj)
      return false;

  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = kNumRegressionMasks;
  cv::Mat box_mask(full_label_height,
                   full_label_width, CV_32F,
                   cv::Scalar(-1.0));
  cv::Mat poly_mask(full_label_height,
                   full_label_width, CV_32F,
                   cv::Scalar(-1.0));
  vector<bool> has_poly_mask(data.car_boxes_size()+1, false);
  cv::Mat ellipse_mask(full_label_height,
                   full_label_width, CV_32F,
                   cv::Scalar(-1.0));

  // masks
  cv::Mat sbox_mask(full_label_height,
                   full_label_width, CV_8U,
                   cv::Scalar(255));

  int crop_img_w = param.cropped_width();
  int crop_img_h = param.cropped_height();
  cv::Mat box_mask_origin(crop_img_h, crop_img_w,
                  CV_32F, cv::Scalar(-1.0));
  cv::Mat poly_mask_origin(crop_img_h, crop_img_w,
                  CV_32F, cv::Scalar(-1.0));
  cv::Mat ellipse_mask_origin(crop_img_h, crop_img_w,
                  CV_32F, cv::Scalar(-1.0));
  cv::Mat dis_inside(crop_img_h, crop_img_w,
                  CV_32F, cv::Scalar(INF_DIS));
  cv::Mat dis_outside(crop_img_h, crop_img_w,
                  CV_32F, cv::Scalar(INF_DIS));

  cv::Mat debug_boundary(crop_img_h, crop_img_w,
                  CV_8U, cv::Scalar(255));

  vector<bool> has_ellipse_mask(data.car_boxes_size()+1, false);
  vector<int> itypes(data.car_boxes_size()+1);

  //cv::Mat circle_mask, poly_mask;
  vector<cv::Mat *> labels;                  // finaly, these are the labels -hlhe
  for (int i = 0; i < num_total_labels; ++i)
  {
    labels.push_back(new cv::Mat(full_label_height,
                                 full_label_width,
                                 CV_32F,
                                 cv::Scalar(0.0)));
  }

  int hlhe_valid_bbox_cnt = 0;
  for (int i = 0; i < data.car_boxes_size(); ++i)
  {
    float xmin = data.car_boxes(i).xmin()*resize;
    float ymin = data.car_boxes(i).ymin()*resize;
    float xmax = data.car_boxes(i).xmax()*resize;
    float ymax = data.car_boxes(i).ymax()*resize;
    int ttype = data.car_boxes(i).type();
    itypes[i] = ttype;
    CHECK_LT(ttype+1, param.catalog_number());

    float ow = xmax - xmin;  // width after resize -hlhe
    float oh = ymax - ymin;
    xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
    xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
    ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
    ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());

    float w = xmax - xmin;
    float h = ymax - ymin;

    // drop boxes that unrecognize               
    if (w*h < ow*oh*unrecog_factor){    // droped here(h=0) -hlhe
      continue;
    }
    if (w < 4 || h < 4) {  // drop boxes that are too small
      continue;
    }

    if (std::max(w,h) < param.reco_min() || std::max(w,h) > param.reco_max()){
      continue;
    }

    // shrink bboxes
    int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);  // gmin must < 640 -hlhe
    int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);   // gmax must < 480 -hlhe
    int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
    int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    if (gxmin >= full_label_width)   // full_label_width=160 -hlhe
    {
      gxmin = full_label_width - 1;
    }
    if (gymin >= full_label_height)
    {
      gymin = full_label_height - 1;
    }
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, full_label_width);
    CHECK_LE(gymax, full_label_height);

    // gxmin,fxmax -> mask grid range -hlhe
    // deal with label critical conditions -hlhe
    if (gxmin == gxmax)
    {
      if (gxmax < full_label_width - 1)
      {
        gxmax++;
      }
      else if (gxmin > 0)
      {
        gxmin--;
      }
    }

    if (gymin == gymax)
    {
      if (gymax < full_label_height - 1)
      {
        gymax++;
      }
      else if (gymin > 0)
      {
        gymin--;
      }
    }
    CHECK_LT(gxmin, gxmax);
    CHECK_LT(gymin, gymax);
    if (gxmax == full_label_width)
    {
      gxmax--;
    }
    if (gymax == full_label_height)
    {
      gymax--;
    }
 
    // boxes that remain
    cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);
    bb_pushback(bbox_pts, gxmin, gxmax, gymin, gymax);

    // calc dis
    float big_box_shrink = 1.3;
    float small_box_shrink = 0.5;
    vector<int> big_box = get_scaled_box(xmin, ymin, xmax, ymax, 1.0,   big_box_shrink, crop_img_w, crop_img_h);
    vector<int> small_box = get_scaled_box(xmin, ymin, xmax, ymax, 1.0, small_box_shrink, crop_img_w, crop_img_h);
    vector<int> ori_box = get_scaled_box(xmin, ymin, xmax, ymax, 1.0, 1.0, crop_img_w, crop_img_h);

    cv::Rect ori_rect(ori_box[0], ori_box[1], ori_box[2]-ori_box[0]+1, ori_box[3]-ori_box[1]+1);
    cv::Scalar idc(i+1);
    cv::rectangle(box_mask_origin, ori_rect, idc, -1);

    float flabels[num_total_labels] = {1.0f,
                                       (float)xmin,
                                       (float)ymin,
                                       (float)xmax,
                                       (float)ymax,
                                       1.0f / w,
                                       1.0f / h,
                                       1.0f};

    for (int j = 0; j < num_total_labels; ++j)
    {
      cv::Mat roi(*labels[j], r);       // get a patch/roi(Just a matrix header) -hlhe
      roi = cv::Scalar(flabels[j]);     // fill the roi!!!(reference) -hlhe
    }

    cv::Scalar idcolor(i);    // add hlhe
    cv::rectangle(box_mask, r, idcolor, -1);

    if (param.use_mask())
    {
        draw_mask(data, poly_mask_origin, ellipse_mask_origin, i, w_off, h_off, 1.0, resize);
        
        caffe::CarBoundingBox box = data.car_boxes(i);
        if (box.poly_mask_size()>2)
        {
            std::vector<cv::Point2i> pts;
            for (int j=0; j<box.poly_mask_size(); j++)
            {
                caffe::FixedPoint p = box.poly_mask(j);
                pts.push_back(cv::Point2i((int) ((p.x()*resize-w_off)*scaling),
                                          (int) ((p.y()*resize-h_off)*scaling)));
            }
            std::vector< std::vector<cv::Point2i> > ptss;
            ptss.push_back(pts);
            cv::fillPoly(poly_mask, ptss, idcolor);
            has_poly_mask[i] = true;
        }
        if (box.ellipse_mask_size() > 2)
        {
            std::vector<cv::Point2i> pts;
            for (int j=0; j<box.ellipse_mask_size(); j++)
            {
                caffe::FixedPoint p = box.ellipse_mask(j);
                pts.push_back(cv::Point2i((int) ((p.x()*resize-w_off)*scaling),
                                          (int) ((p.y()*resize-h_off)*scaling)));
            }
            int tmp=0;
            while (pts.size()<5)
                pts.push_back(pts[tmp++]);

            cv::RotatedRect rbox = cv::fitEllipse(pts);
            cv::ellipse(ellipse_mask, rbox, idcolor, -1);
            has_ellipse_mask[i] = true;
        }
    }

    //a.Find boudary points
    vector<vector<int> > boundary_pts_vec;
    find_instance_boundary_pts(boundary_pts_vec, ori_box, box_mask_origin,
        poly_mask_origin, ellipse_mask_origin, has_poly_mask[i], has_ellipse_mask[i]);
    
    // for( int y=0; y<box_mask_origin.rows; y++){
    //     for( int x=0; x<box_mask_origin.cols; x++){
    //         float v1 = box_mask_origin.at<float>(y,x);
    //         int v2 = int(poly_mask_origin.at<float>(y,x));
    //         int v3 = int(ellipse_mask_origin.at<float>(y,x));
    //         if( has_poly_mask[i] ){
    //             v1 = v1 * int(v2>-1);
    //         }
    //         if( has_ellipse_mask[i] ){
    //             v1 = v1 * int(v3>-1);
    //         }

    //         if( v1>0 ){
    //             debug_boundary.at<unsigned char>(y,x) = 64;
    //         }
    //     }
    // }
    // for( int k=0; k<boundary_pts_vec.size(); k++){
    //     int y = boundary_pts_vec[k][1];
    //     int x = boundary_pts_vec[k][0];
    //     debug_boundary.at<unsigned char>(y,x) = 128;
    // }

    //LOG(INFO)<<"boundary_pts_vec.size() "<<boundary_pts_vec.size();

    // b.Calc distance
    float obj_size = std::min(ori_box[3]-ori_box[1], ori_box[2]-ori_box[0]);
    //LOG(INFO)<<"obj_size:"<<obj_size;
    if( boundary_pts_vec.size()>0 ){
        calc_distance_to_boundary(big_box, small_box, obj_size, i+1, boundary_pts_vec, box_mask_origin, 
          poly_mask_origin, ellipse_mask_origin, dis_inside, dis_outside, has_poly_mask[i], has_ellipse_mask[i]);
    }

    hlhe_valid_bbox_cnt += 1; //hlhe
  }
   cv::resize(dis_inside, dis_inside,  cv::Size(full_label_width, full_label_height), 0,0, CV_INTER_LINEAR);
   cv::resize(dis_outside, dis_outside, cv::Size(full_label_width, full_label_height), 0,0, CV_INTER_LINEAR);
//    cv::resize(debug_boundary, debug_boundary, cv::Size(full_label_width, full_label_height) );

//   // //show dis_inside,dis_outside
//   float max_dis = 16;
//   CHECK_EQ(dis_inside.rows, debug_boundary.rows);
//   CHECK_EQ(dis_inside.cols, debug_boundary.cols);
//   CHECK_EQ(dis_outside.rows, debug_boundary.rows);
//   CHECK_EQ(dis_outside.cols, debug_boundary.cols);
//   for( int y=0; y<debug_boundary.rows; y++){
//     for( int x=0; x<debug_boundary.cols; x++){
//         float v = dis_inside.at<float>(y,x);
//         if( v>max_dis ){
//             v = max_dis;
//         }
//         if( v<0 ){
//             v = 0;
//         }
//         v = int(v/max_dis * 255);
//         debug_boundary.at<unsigned char>(y,x) = v;
//     }
//   }

//   strstream ss;
//   string save_root = "/data2/HongliangHe/work2017/TrafficSign/node113/boundary_ignore/my/0518_boundary_ignore/debug_imgs/";
//   string rnd_name = "";
//   string save_name = "";
//   ss<<Rand();
//   ss>>rnd_name;
//   save_name = save_root + rnd_name + "_bdy.jpg";
//   imwrite(save_name.c_str(), debug_boundary);
//   LOG(INFO)<<save_name<<" saved!";

  //============== init ===============
  datum->set_channels(num_total_labels);
  datum->set_height(full_label_height);
  datum->set_width(full_label_width);
  datum->set_label(0);  // dummy label
  datum->clear_data();
  datum->clear_float_data();

  int total_num_pixels = 0;  // Count label pixels, to get pixel label weight -hlhe
  for (int y = 0; y < full_label_height; ++y)
  {
    for (int x = 0; x < full_label_width; ++x)
    {
        float &val = box_mask.at<float>(y,x);
        if (val != -1)
        {
            int id = (int)val;
            if (has_poly_mask[id] && poly_mask.at<float>(y,x) != val)
                val = -1;
            if (has_ellipse_mask[id] && ellipse_mask.at<float>(y,x) != val)
                val = -1;
            float valo = labels[0]->at<float>(y, x);
            if (valo == 0)
                val = -1;
            if (val != -1)
                total_num_pixels++;
        }
    }
  }

  // normalization
  if (total_num_pixels != 0) {
    float reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        if (box_mask.at<float>(y,x) == -1)
            labels[num_total_labels - 1]->at<float>(y, x) = 0.0f;
        else
            labels[num_total_labels - 1]->at<float>(y, x) = reweight_value;
      }
    }
  }

 // assignments
  for (int m = 0; m < num_total_labels; ++m)
  {
    for (int y = 0; y < full_label_height; ++y)
    {
      for (int x = 0; x < full_label_width; ++x)
      {
        float adjustment = 0;
        float val = labels[m]->at<float>(y, x);
        if (m == 0 || m > 4)  // not bbox coordinates -hlhe
        {
            if (m == 0 && param.use_mask() && val == 1.0f)
            {
                val = (box_mask.at<float>(y,x)==-1)?0.0f:1.0f;  // mask fix -hlhe
            }
        }
        else if (labels[0]->at<float>(y, x) == 0.0)  // Have not mask label -hlhe
        {
          // do nothing
        }
        else if (m % 2 == 1)
        {
          // x coordinate              // minus "adjustment", then get offset?? -hlhe
          adjustment = x / scaling;    // Get right coordinate target -hlhe
        }
        else
        {
          // y coordinate
          adjustment = y / scaling;
        }
        datum->add_float_data(val - adjustment);
      }
    }
  }

  // assign distance
  Dtype* pdis = lab_dis_vec.at(0);
  assert(pdis!=NULL);

  for( int y=0; y<dis_inside.rows; y++){
      for( int x=0; x<dis_inside.cols; x++){
          int idx = y*dis_inside.cols + x;
          pdis[idx] = (Dtype)dis_inside.at<float>(y,x);
      }
  }
  int inner_num = dis_outside.rows * dis_outside.cols;
  for( int y=0; y<dis_outside.rows; y++){
      for( int x=0; x<dis_outside.cols; x++){
          int idx = y*dis_inside.cols + x + inner_num;
          pdis[idx] = (Dtype)dis_outside.at<float>(y,x);
      }
  }

  CHECK_EQ(datum->float_data_size(),
           num_total_labels * full_label_height * full_label_width);

  for (int i = 0; i < num_total_labels; ++i)
  {
    delete labels[i];
  }
  return have_obj;
}

float get_box_size(const caffe::CarBoundingBox &box) {
    float xmin = box.xmin();
    float ymin = box.ymin();
    float xmax = box.xmax();
    float ymax = box.ymax();
    return std::max(xmax-xmin, ymax-ymin);
}

// This function is called on prefetch thread
template<typename Dtype>
void DriveDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
//  LOG(INFO)<<"Get in load_batch()!";
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data().count());
  DriveDataParameter param = this->layer_param_.drive_data_param();

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  DrivingData data;
  string *raw_data = NULL;

  vector<int> top_shape(4,0);
  int shape[4] = { batch_size,                 // batch size -hlhe
                   3,                          // channel
                  (int)param.cropped_height(), // 480
                  (int)param.cropped_width()}; // 640
  memcpy(&top_shape[0], shape, sizeof(shape)); // image data shape

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data().Reshape(top_shape);    // batch->data image batch -hlhe

  Dtype* top_data = batch->data().mutable_cpu_data();
  const Dtype* data_mean_c = this->data_transformer_->data_mean_.cpu_data();
  vector<Dtype> v_mean(data_mean_c, data_mean_c+this->data_transformer_->data_mean_.count());
  Dtype *data_mean = &v_mean[0];       // batch mean pictures(vector) -hlhe

  vector<Dtype*> top_labels;
  Dtype *label_type = NULL;
  Dtype *label_mask = NULL;           // 3 path masks
  Dtype *label_dis = NULL;
  int mask_one_batch_count = 0;
  int lab_dis_batch_num = 0;

  if (this->output_labels_)           // generate labels?? -hlhe
  {
    Dtype *top_label = batch->label().mutable_cpu_data();
    top_labels.push_back(top_label);
    label_mask = batch->label(1).mutable_cpu_data();
    label_dis  = batch->label(2).mutable_cpu_data();  // to store distance to boundary 2017/05/15
    mask_one_batch_count = batch->label(1).shape()[1] * batch->label(1).shape()[2] * batch->label(1).shape()[3];
    label_type = batch->label(3).mutable_cpu_data();  // commented by hlhe
    const vector<int> S = batch->label(2).shape();
    lab_dis_batch_num = S[1] * S[2] * S[3];
  }

  const int crop_num = this->layer_param().drive_data_param().crop_num();
  int type_label_strip = param.tiling_height()*param.tiling_width()            // ?? -hlhe
                        *param.catalog_resolution()*param.catalog_resolution();
  bool need_new_data = true;
  int bid = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    timer.Start();
    // get a datum
    if (need_new_data)
    {
        data.ParseFromString(*(raw_data = this->reader_.full().pop("Waiting for data")));
        bid = 0;
        need_new_data = false;
        //hlhe_draw_bbs(data);
    }
    else
    {
        bid ++;
    }
    if (item_id+1 == batch_size)
        need_new_data = true;

    read_time += timer.MicroSeconds();
    timer.Start();
    Dtype *t_lable_type = NULL;
    if (label_type != NULL)
    {
        t_lable_type = label_type + type_label_strip*item_id;
        caffe_set(type_label_strip, (Dtype)0, t_lable_type);
    }

    // data <- read from MDB -hlhe
    vector<Datum> label_datums(kNumLabels);           // write label to -> label_datums -hlhe
    const Datum& img_datum = data.car_image_datum();
    const string& img_datum_data = img_datum.data();  // raw string, read from mdb -hlhe

    // random_crop_ratio: 0.1 -hlhe
    float rnd_th = this->layer_param().drive_data_param().random_crop_ratio();
    if( false==lab_auxi_.is_normal_status() )
    {
        rnd_th = this->layer_param().drive_data_param().hm_random_crop_ratio();
    }

    bool can_pass = rand_float() > rnd_th;
    int cheight = param.cropped_height();    // cheight->crop_height
    int cwidth = param.cropped_width();      // cwidth ->crop_width
    int channal = img_datum.channels();

    int hid = bid/crop_num;
    float bsize = get_box_size(data.car_boxes(hid));  // bsize->max box size -hlhe
    float rmax = std::min(param.train_max() / bsize,  // min(train_box_max/box_size, resize_max)
                          param.resize_max());

    float rmin = std::max(param.train_min() / bsize,  // rmax->resize_max??
                          param.resize_min());        // rmin->resize_min?? -hlhe

    // TODO : hard code type
    vector<cv::Mat> picgen;
    if (genpic.size()>0)
    {
        for (int i=0; i<4; i++)
            picgen.push_back(cv::Mat::zeros(cheight, cwidth, CV_32F));
    }

    int try_again_count = 0;
    int MAX_TRY_AGAIN = 100;

try_again:
    try_again_count += 1;
    if ( try_again_count>MAX_TRY_AGAIN )
    {
        can_pass = false;
    }

    float resize = this->layer_param().drive_data_param().resize();
    int h_off, w_off;
    if (resize < 0)
    {
        if (rmax <= rmin || !can_pass)
        {
            can_pass = false;
            resize = rand_float() * (param.resize_max()-param.resize_min()) + param.resize_min();
            int rheight = (int)(img_datum.height() * resize);
            int rwidth = (int)(img_datum.height() * resize);
            h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
            w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth);
        }
        else
        {
            //can_pass = false;
            int h_max, h_min, w_max, w_min;
            resize = rand_float() * (rmax-rmin) + rmin;  // random resize scale -hlhe

            //LOG(INFO) << "resize " << rmax << " " << rmin << " " << resize << ' ' << item_id;
            h_min = data.car_boxes(hid).ymin()*resize - param.cropped_height();
            h_max = data.car_boxes(hid).ymax()*resize;
            w_min = data.car_boxes(hid).xmin()*resize - param.cropped_width();
            w_max = data.car_boxes(hid).xmax()*resize;
            w_off = (Rand() % (w_max-w_min) + w_min);   // off range ??
            h_off = (Rand() % (h_max-h_min) + h_min);
            //w_off = data.car_boxes(hid).xmax()*resize - param.cropped_width();;
            //h_off = data.car_boxes(hid).ymax()*resize - param.cropped_height();
        }
    }
    else
    {
        // parameter "resize" is fixed! -hlhe
        int rheight = (int)(img_datum.height() * resize);  // resize_height -hlhe
        int rwidth = (int)(img_datum.height() * resize);

        h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
        w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth);
    }
    //LOG(INFO) << "?" << w_off << ' ' << h_off << ' ' << resize << ' ' << rmax << ' ' << rmin;

    vector<Dtype*> label_dis_vec;
    vector<vector<int> > bb_pts;
    vector<Dtype*> mask_muta_data;
    int batch_pos = item_id * mask_one_batch_count;
    mask_muta_data.push_back(label_mask+batch_pos); // output pointers
    label_dis_vec.push_back(label_dis + item_id*lab_dis_batch_num);

    if (this->output_labels_)
    {
      bool have_obj = false;
      bool is_normal = lab_auxi_.is_normal_status();
      vector<vector<int> > bbox_pts;
      have_obj = ReadBoundingBoxLabelToDatum(data,                 // raw data read from mdb
                                              &label_datums[0],     // output
                                              h_off,                // random offset
                                              w_off,                // random offset
                                              resize,               //
                                              param,                // DriveDataParameter, maybe read from net prototxt
                                              (float*)t_lable_type, // output buf??
                                              can_pass,             // random pass flag
                                              hid,                  // bbox idx ??
                                              genpic,
                                              genpic_type,
                                              picgen,
                                              mask_muta_data,
                                              bb_pts,
                                              is_normal,
                                              label_dis_vec);

      if (!have_obj) // for image that has no objects
      {
          if ( can_pass )
          {
            goto try_again;
          }
      }
    }

    if (try_again_count>=MAX_TRY_AGAIN)
    {
        //LOG(INFO)<<"try_again_count:"<<try_again_count;
    }

    vector<Batch<Dtype>*> batch_vec;
    batch_vec.push_back(batch);

    lab_auxi_.gen_gray_zone(bb_pts);
    lab_auxi_.apply_gray_zone(label_datums[0], 0, Lab4ClsChanelIdx,
                              batch_vec, item_id);

    if (bid+1 >= crop_num*data.car_boxes_size())
        need_new_data = true;

    // image data minus mean value
    cv::Mat_<Dtype> mean_img(cheight, cwidth);
    Dtype* itop_data = top_data+item_id*channal*cheight*cwidth;
    float mat[] = { 1.f*resize,0.f,(float)-w_off, 0.f,1.f*resize,(float)-h_off };
    cv::Mat_<float> M(2,3, mat);

    for (int c=0; c<img_datum.channels(); c++) 
    {
        cv::Mat_<Dtype> crop_img(cheight, cwidth, itop_data+c*cheight*cwidth);
        cv::Mat_<Dtype> pmean_img(img_datum.height(), img_datum.width(),
                          data_mean + c*img_datum.height()*img_datum.width());
        cv::Mat p_img(img_datum.height(), img_datum.width(), CV_8U,
               ((uint8_t*)&(img_datum_data[0])) + c*img_datum.height()*img_datum.width());
        p_img.convertTo(p_img, crop_img.type());
        cv::warpAffine(pmean_img, mean_img, M, mean_img.size(), cv::INTER_CUBIC);
        cv::warpAffine(p_img, crop_img, M, crop_img.size(), cv::INTER_CUBIC);

        crop_img -= mean_img;
        if (picgen.size()) 
        {
            cv::multiply(crop_img, 1-picgen[3], crop_img);
            picgen[c] = (picgen[c] - 255./2)*param.gen_scale();
            cv::multiply(picgen[c], picgen[3], picgen[c]);
            crop_img += picgen[c];
            //picgen[c].copyTo(crop_img);
            //crop_img = picgen[c];
            //crop_img = crop_img*(1-picgen[3]) + picgen[c]*picgen[3];
        }
        crop_img *= this->layer_param().drive_data_param().scale();
    }

    // Copy label.
    if (this->output_labels_)
    {
      for (int i = 0; i < kNumLabels; ++i)
      {
        for (int c = 0; c < label_datums[i].channels(); ++c)
        {
          for (int h = 0; h < label_datums[i].height(); ++h)
          {
            for (int w = 0; w < label_datums[i].width(); ++w)
            {
              //const int top_index = ((item_id * label_datums[i].channels() + c)
              //                   * label_datums[i].height() + h) * label_datums[i].width() + w;
              const int top_index = ((item_id*batch->label().channels() + c)
                                     *batch->label().height() + h)*batch->label().width() + w;

              const int data_index = (c * label_datums[i].height() + h) * \
                label_datums[i].width() + w;
              float label_datum_elem = label_datums[i].float_data(data_index);
              top_labels[i][top_index] = static_cast<Dtype>(label_datum_elem);
            }
          }
        }
      }
    }
    trans_time += timer.MicroSeconds();

    if (need_new_data) {
        this->reader_.free().push(const_cast<string*>(raw_data));
    }
  }

  lab_auxi_.counter_tic(); //�����Լ���.����Ҫ����һ��ȫ�ֱ����������źŵ�ͬ��.
  lab_auxi_.set_global_status(g_mining_status,
            g_NormalStatus, g_HardMiningStatus);
  timer.Stop();
  batch_timer.Stop();
  //LOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  //LOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  //LOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

int LabelAuxiliary::gen_gray_zone(vector<vector<int> >& bbs_pts)
{
    // ��ͨ����,��������
    if( is_normal_status() )
    {
        return -1;
    }

    reset_gray_zone(-1, -1);
    for(int bb_idx=0; bb_idx<bbs_pts.size(); bb_idx++)
    {
        int x1 = bbs_pts[bb_idx][0];
        int x2 = bbs_pts[bb_idx][1];
        int y1 = bbs_pts[bb_idx][2];
        int y2 = bbs_pts[bb_idx][3];

        float dw = (x2-x1-1) * (bbox_sacle_-1)/2.0;
        float dh = (y2-y1-1) * (bbox_sacle_-1)/2.0;
        dw = std::max<float>(dw, 2.0);
        dh = std::max<float>(dh, 2.0);

        int new_x1 = x1 - int(dw+0.5);
        int new_x2 = x2 + int(dw+0.5);
        int new_y1 = y1 - int(dh+0.5);
        int new_y2 = y2 + int(dh+0.5);

        if (new_x1<0)
            new_x1 = 0;
        if (new_x2>=gray_zone_.cols)
            new_x2 = gray_zone_.cols-1;
        if (new_y1<0)
            new_y1 = 0;
        if (new_y2>=gray_zone_.rows)
            new_y2 = gray_zone_.rows-1;

        for( int i=new_x1; i<=new_x2; i++)
        {
            for( int j=new_y1; j<=new_y2; j++)
            {
                gray_zone_.at<float>(j,i) = GRAY_ZONE_LAB;
            }
        }
    }

    return 0;
}

// ������ 2016/09/12 22:20
int LabelAuxiliary::update_mask(cv::Mat& ori_mask)
{
    // ��ͨ����,��������
    if( is_normal_status() )
    {
        return -1;
    }

    int gs = 3;                     //�ṹԪ��(�ں˾���)�ĳߴ�
    cv::Mat mask = ori_mask>INVALIDE_BBOX_IDX;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                      cv::Size(gs, gs),
                                      cv::Point(-1,-1));

    //my_show_mat("ori_mask", (ori_mask!=INVALIDE_BBOX_IDX)*255);
    cv::erode(mask, mask, element);
    //my_show_mat("mask", mask*125);

    //��ʱò����-1��ʾ����
    for(int i=0; i<ori_mask.rows; i++)
    {
        for(int j=0; j<ori_mask.cols; j++)
        {
            if( mask.at<uchar>(i,j)==0 )  //��ʴ֮��,ȡֵΪ[0,255]
            {
                ori_mask.at<float>(i,j) = INVALIDE_BBOX_IDX;
            }
        }
    }
    //my_show_mat("after_mask", (ori_mask!=INVALIDE_BBOX_IDX)*250);

    return 0;
}


// ����ǰ�õ���gray_zone���ӵ�label��ȥ
template<typename Dtype>
int LabelAuxiliary::apply_gray_zone(Datum& lab_datum, int ori_ch, int tar_ch,
                                    vector<Batch<Dtype>*>& my_batch, int batch_idx)
{
    int h = gray_zone_.rows;
    int w = gray_zone_.cols;
    assert(lab_datum.height()==h);
    assert(lab_datum.width() ==w);

    //my_show_mat("gray_zone", gray_zone_*125);
    Dtype* batch_label = my_batch[0]->label().mutable_cpu_data();
    cv::Mat vis = cv::Mat::zeros(h, w, CV_32FC1);


    if ( is_normal_status() )
    {
        for( int i=0; i<h; i++)
        {
            for( int j=0; j<w; j++)
            {
                int ori_idx = (ori_ch*lab_datum.height()+i)*lab_datum.width() + j;
                Dtype lab_val = lab_datum.float_data(ori_idx);

                // ֱ��д�뵽batch��label��
                int tar_index = ((batch_idx * my_batch[0]->label().channels() + tar_ch)
                                  *my_batch[0]->label().height() + i)*my_batch[0]->label().width() + j;
                batch_label[tar_index] = lab_val;
                vis.at<float>(i,j) = lab_val * 125;
            }
        }
    }
    else  // �к�������������
    {
        for( int i=0; i<h; i++)
        {
            for( int j=0; j<w; j++)
            {
                int ori_idx = (ori_ch*lab_datum.height()+i)*lab_datum.width() + j;

                Dtype lab_val = lab_datum.float_data(ori_idx);
                float gray_zone_val = gray_zone_.at<float>(i,j);
                Dtype new_val = BACKGROUND_LAB;

                if( lab_val!=TRAFFIC_SIGN_LAB &&
                    gray_zone_val == GRAY_ZONE_LAB)
                {
                    new_val = GRAY_ZONE_LAB;
                }
                else
                {
                    new_val = lab_val;
                }

                // ֱ��д�뵽batch��label��
                int tar_index = ((batch_idx * my_batch[0]->label().channels() + tar_ch)
                                  *my_batch[0]->label().height() + i)*my_batch[0]->label().width() + j;
                batch_label[tar_index] = new_val;
                vis.at<float>(i,j) = new_val * 125;
            }
        }
    }

    //my_show_mat("vis", vis);
    return 0;
}

int LabelAuxiliary::reset_gray_zone(int h, int w)
{
    if(has_inited_)
    {
        gray_zone_.setTo(BACKGROUND_LAB);
    }
    else
    {
        assert(h>0);
        assert(w>0);
        gray_zone_ = cv::Mat::zeros(h, w, CV_32F);
    }

    has_inited_ = true;
    return 0;
}


int MaskCorroder::init(int h, int w)
{
    tmp_mask1_ = cv::Mat::zeros(h,w, CV_32FC1);
    tmp_mask2_ = cv::Mat::zeros(h,w, CV_32FC1);
    return 0;
}

// ÿ��mask�����ɷ�ʽ������һ��!
int MaskCorroder::process(cv::Mat& tar_mask, PointsVec my_ptss, int tar_val)
{
    //my_show_mat("before erode", tar_mask);
    tmp_mask1_.setTo(0.0);
    //tmp_mask2_.setTo(0);

    cv::Scalar bg(0.0);
    cv::fillPoly(tmp_mask1_, my_ptss, bg);

    int gs = 3; //�ṹԪ��(�ں˾���)�ĳߴ�
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
                                      cv::Size(gs, gs),
                                      cv::Point(-1,-1));
    cv::erode(tmp_mask1_, tmp_mask2_, element);

    for(int i=0; i<tmp_mask1_.rows; i++)
    {
        for(int j=0; j<tmp_mask1_.cols; j++)
        {
            if(tmp_mask1_.at<uchar>(i,j)>0)
            {
                if(tmp_mask2_.at<uchar>(i,j)>0)
                    tar_mask.at<float>(i,j) = tar_val;
            }
        }
    }
    //my_show_mat("after erode", tar_mask);

    return 0;
}
INSTANTIATE_CLASS(DriveDataLayer);
REGISTER_LAYER_CLASS(DriveData);

}  // namespace caffe
