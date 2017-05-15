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


namespace caffe {

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
  //һ��batch����,ÿ����hm_perid����һ�������ĵ���
  int hm_perid = this->layer_param().drive_data_param().hm_perid();
  float ign_bb_scale = this->layer_param().drive_data_param().ignore_bbox_scale();
  lab_auxi_.set_perid(hm_perid);
  lab_auxi_.set_bbox_scale(ign_bb_scale);
  srand(time(0));
  const int batch_size = this->layer_param_.data_param().batch_size();
  DriveDataParameter param = this->layer_param().drive_data_param();
  // Read a data point, and use it to initialize the top blob.

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
      LOG(INFO) << "using genlist " << param.genlist_file();
      std::ifstream fin(param.genlist_file().c_str());
      std::string picname;
      while (fin >> picname) {
          cv::Mat img = cv::imread(picname.c_str(), CV_LOAD_IMAGE_UNCHANGED);
          int tp;
          fin >> tp;
          DLOG(INFO) << "get " << picname << ' ' << tp;
          genpic.push_back(img);
          genpic_type.push_back(tp);
      }
      LOG(INFO) << "total " << genpic.size() << " artificial pics";
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

    CHECK_GE(top.size(), 3);
    top[2]->Reshape(mask_shape);                  // Because there has no type loss. -hlhe

    for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    {
      this->prefetch_[i].label(0).Reshape(label_shape);
      this->prefetch_[i].label(1).Reshape(mask_shape);
      this->prefetch_[i].label(2).Reshape(type_shape);
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
        bool normal_status)              // ��ͨ״̬/Hard mining״̬
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

  // detection and classification have different resolutions -hlhe
  const int type_label_width = width * param.catalog_resolution();
  const int type_label_height = height * param.catalog_resolution();
  const int type_stride = full_label_width / type_label_width;


  // fast check
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

#define MASK_ORI_CHN       0
#define GEN_MASK1_MASK2
#ifdef GEN_MASK1_MASK2
    cv::Mat path1_mask(full_label_height,
                       full_label_width,
                       CV_32F,
                       cv::Scalar(BACKGROUND_LAB));

    cv::Mat path2_mask(full_label_height,
                       full_label_width,
                       CV_32F,
                       cv::Scalar(BACKGROUND_LAB));

    cv::Mat path3_mask(full_label_height,
                       full_label_width,
                       CV_32F,
                       cv::Scalar(BACKGROUND_LAB));
#endif

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
    if (w*h < ow*oh*unrecog_factor)    // droped here(h=0) -hlhe
        continue;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }

    if (std::max(w,h) < param.reco_min() || std::max(w,h) > param.reco_max())
        continue;

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

    // ���յĸ���Ȥ����
    cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);
    bb_pushback(bbox_pts, gxmin, gxmax, gymin, gymax);

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

// generate mask1
// 32~64 64~108 108~180
#ifdef GEN_MASK1_MASK2
    float rec_rate_th = 0.8;
    int gt_res = 4;
    int half_m1_concep_f = PATH1_RECIPTION_FILD / gt_res / 2;
    int half_m2_concep_f = PATH2_RECIPTION_FILD / gt_res / 2;

    int m1_bb_xmin = int((xmin+0.5)/gt_res);
    int m1_bb_xmax = int((xmax+0.5)/gt_res);
    int m1_bb_ymin = int((ymin+0.5)/gt_res);
    int m1_bb_ymax = int((ymax+0.5)/gt_res);
    m1_bb_xmax = std::min<int>(m1_bb_xmax, full_label_width-1);
    m1_bb_ymax = std::min<int>(m1_bb_ymax, full_label_height-1);

    int bb_w = m1_bb_xmax-m1_bb_xmin;
    int bb_h = m1_bb_ymax-m1_bb_ymin;
    int bb_size = std::max<int>(xmax-xmin, ymax-ymin);
    //LOG(INFO)<<"bb_size:"<<bb_size;

    // ע������bb_size��λ��
    if( !normal_status )
    {
        rescale_bb(m1_bb_xmin,m1_bb_xmax,
                   m1_bb_ymin,m1_bb_ymax,
                   full_label_height,
                   full_label_width, 1.4f);
    }

    for (int x_pos=m1_bb_xmin; x_pos<=m1_bb_xmax; x_pos++)
    {
        for(int y_pos=m1_bb_ymin; y_pos<=m1_bb_ymax; y_pos++)
        {
            int concep_xmin = x_pos - half_m1_concep_f;
            int concep_xmax = x_pos + half_m1_concep_f;
            int concep_ymin = y_pos - half_m1_concep_f;
            int concep_ymax = y_pos + half_m1_concep_f;

            path1_mask.at<float>(y_pos,x_pos) = GRAY_ZONE_LAB;
            path2_mask.at<float>(y_pos,x_pos) = GRAY_ZONE_LAB;
            path3_mask.at<float>(y_pos,x_pos) = GRAY_ZONE_LAB;

            // path1 mask
            if(bb_size<=PATH1_MAX_OBJ_SIZE)
            {
                float intersec = (float)calc_cross(m1_bb_xmin,m1_bb_xmax,
                                                   m1_bb_ymin,m1_bb_ymax,
                                                 concep_xmin,concep_xmax,
                                                 concep_ymin,concep_ymax);
                float rate = intersec / (bb_w*bb_h);
                if (rate > rec_rate_th)
                {
                    if(normal_status){
                        path1_mask.at<float>(y_pos,x_pos) = TRAFFIC_SIGN_LAB;  // target!
                    }
                    else {   // ���Ǳ�־��Χ��Ҫ���Ե�����,���������߶ȶ����ɵĺ���
                        path1_mask.at<float>(y_pos,x_pos) = IGNORE_EDGE_LAB;
                    }
                }
            }
            else if(bb_size>PATH3_MIN_OBJ_SIZE)
            {
                path1_mask.at<float>(y_pos,x_pos) = BACKGROUND_LAB;
            }

            // path2 mask
            concep_xmin = x_pos - half_m2_concep_f;
            concep_xmax = x_pos + half_m2_concep_f;
            concep_ymin = y_pos - half_m2_concep_f;
            concep_ymax = y_pos + half_m2_concep_f;
            if (bb_size<=PATH2_MAX_OBJ_SIZE && bb_size>PATH2_MIN_OBJ_SIZE)
            {
                float intersec = (float)calc_cross(m1_bb_xmin,m1_bb_xmax,
                                                   m1_bb_ymin,m1_bb_ymax,
                                                 concep_xmin,concep_xmax,
                                                 concep_ymin,concep_ymax);
                float rate = intersec / (bb_w*bb_h);
                if (rate > rec_rate_th)
                {
                    if( normal_status )
                    {
                        path2_mask.at<float>(y_pos,x_pos) = TRAFFIC_SIGN_LAB;  // target!
                    }
                    else
                    {   // ���Ǳ�־��Χ��Ҫ���Ե�����,���������߶ȶ����ɵĺ���
                        path2_mask.at<float>(y_pos,x_pos) = IGNORE_EDGE_LAB;
                    }
                }
            }

            // path3 mask
            if(bb_size>PATH3_MIN_OBJ_SIZE)
            {
                if( normal_status )
                {
                    path3_mask.at<float>(y_pos,x_pos) = TRAFFIC_SIGN_LAB;  // target!
                }
                else
                {   // ���Ǳ�־��Χ��Ҫ���Ե�����,���������߶ȶ����ɵĺ���
                    path3_mask.at<float>(y_pos,x_pos) = IGNORE_EDGE_LAB;
                }
            }
            else if(bb_size<=PATH1_MAX_OBJ_SIZE)
            {
                path3_mask.at<float>(y_pos,x_pos) = BACKGROUND_LAB;
            }
        }
    }
#endif

    // dirty code copy
    hlhe_valid_bbox_cnt += 1; //hlhe
  }


  // Generate artificial traffic signs -hlhe
  if (genpic.size() > 0 && rand_float() < param.gen_rate())
  {
      // generate artificial mark
      int ip = Rand() % genpic.size();
      cv::Mat img;
      genpic[ip].convertTo(img, CV_32FC4);    // template pic -hlhe
      vector<cv::Mat> imgs(4);
      cv::split(img, imgs);
      imgs[3] /= 255.;

      //===========================================================
      //cv::Mat show_mat(img.rows, img.cols, CV_8UC1);
      //imgs[3].convertTo(show_mat, CV_8UC1);
      //cv::imshow("mask", show_mat); cv::waitKey(-1);
      //===========================================================

      int tp = genpic_type[ip];
      int id = data.car_boxes_size();
      itypes[id] = tp;

      // assume it has 20 pad in image
      int pad = 40;
      int size = 200;
      float nsize = rand_float() * (param.train_max()-param.train_min())+param.train_min();
      float noff_x, noff_y;
      int try_time = 0;
      int max_try = 1000;

      while (try_time++ < max_try)
      {
          noff_x = (rand_float() * (param.cropped_width()-nsize));
          noff_y = (rand_float() * (param.cropped_height()-nsize));
          bool ok = true;

          for (int i = 0; i < data.car_boxes_size(); ++i)
          {
            float xmin = data.car_boxes(i).xmin()*resize - w_off;
            float ymin = data.car_boxes(i).ymin()*resize - h_off;
            float xmax = data.car_boxes(i).xmax()*resize - w_off;
            float ymax = data.car_boxes(i).ymax()*resize - h_off;
            if (check_cross(xmin, xmax, noff_x-pad, noff_x+nsize+pad) &&
                check_cross(ymin, ymax, noff_y-pad, noff_y+nsize+pad))
            {
                ok = false;
                break;
            }
          }
          if (ok) break;
      }

      if (try_time >= max_try)
      {
          LOG(INFO) << "no space to place artificial mark";
      }
      else
      {
          //LOG(INFO) << "can place " << noff_x  << ' ' << noff_y;
          int whs = Rand()&1;
          cv::Point2f pts1[] = {cv::Point2f(0,0), cv::Point2f(1,0), cv::Point2f(0,1)};
          cv::Point2f pts2[] = {cv::Point2f(0,0),
                                cv::Point2f(1-whs*rand_float()*param.shrink_max(),
                                            rand_float()*param.slip_max()*2-param.slip_max()),
                                cv::Point2f(rand_float()*param.slip_max()*2-param.slip_max(),
                                            1-(1-whs)*rand_float()*param.shrink_max())};

          cv::Mat Ms = cv::getAffineTransform(pts1, pts2);
          int blur_size = Rand() % (int)param.blur_max() * 2 + 1;
          float noise_value = rand_float() * param.noise_max();
          float gamma = std::exp(rand_float()*param.gamma_max()*2-param.gamma_max());
          float kk = rand_float()*param.k_max()*2-param.k_max()+1;
          float y0 = rand_float()*param.y0_max()*2-param.y0_max();

          for (int i=0; i<4; i++)
          {
              cv::Mat tmp = cv::Mat::zeros(imgs[0].size(), CV_32FC1);
              cv::warpAffine(imgs[i], tmp, Ms, imgs[i].size(),
                             cv::INTER_CUBIC, cv::BORDER_CONSTANT, i==3?0.0f:255.0f);
              tmp.copyTo(imgs[i]);
              // gaussian blur
              cv::GaussianBlur(imgs[i], tmp, cv::Size(blur_size,blur_size), 0);
              tmp.copyTo(imgs[i]);
              if (i==3) break;

              // add noise
              cv::randn(tmp, 0, 1);
              imgs[i] += (tmp-0.5)*(noise_value*255.);
              double mmin, mmax;
              cv::minMaxLoc(imgs[i],&mmin,&mmax);
              imgs[i] = (imgs[i]-mmin)/(mmax-mmin)*255.0;

              // gamma
              cv::Mat lut_matrix(1, 256, CV_8UC1 );
              uchar * ptr = lut_matrix.ptr();
              for( int j = 0; j < 256; j++ )
              {
                  float v = j / 255.0f;
                  v = pow(v,gamma)*kk+y0;
                  if (v<0) v=0;
                  if (v>1) v=1;
                  ptr[j] = (int)(v * 255.0);
              }
              imgs[i].convertTo(tmp, CV_8UC1);
              cv::LUT( tmp, lut_matrix, tmp );
              tmp.convertTo(imgs[i], CV_32FC1);
              //my_show_mat(tmp);
          }

          float resize = nsize / size;
          noff_x -= pad*resize;
          noff_y -= pad*resize;
          float mat[] = { resize,0.f,noff_x, 0.f,resize,noff_y};
          cv::Mat_<float> M(2,3, mat);
          for (int i=0; i<3; i++)
              cv::warpAffine(imgs[i], picgen[i], M, picgen[0].size(), cv::INTER_CUBIC,
                             cv::BORDER_CONSTANT, 255.0);
          cv::warpAffine(imgs[3], picgen[3], M, picgen[0].size(), cv::INTER_CUBIC,
                 cv::BORDER_CONSTANT, 0.0);
          cv::Mat &mask = picgen[3];
          float xmax=-1e30, xmin=1e30, ymax=-1e30, ymin=1e30;

          for (int y=0; y<mask.size().height; y++)
          {
              for (int x=0; x<mask.size().width; x++)
              {
                  if (mask.at<float>(y,x) > 0.5)
                  {
                      xmax = std::max(xmax, x*1.0f+1);
                      xmin = std::min(xmin, x*1.0f);
                      ymax = std::max(ymax, y*1.0f+1);
                      ymin = std::min(ymin, y*1.0f);
                  }
              }
          }
          float w = xmax-xmin, h = ymax-ymin;

          if (std::max(w,h) >= param.reco_min() && std::max(w,h) <= param.reco_max())
          {
              int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
              int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);
              int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
              int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

              CHECK_LE(gxmin, gxmax);
              CHECK_LE(gymin, gymax);
              if (gxmin >= full_label_width)
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
              if (gxmin == gxmax)
              {
                if (gxmax < full_label_width - 1) {
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
              cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);

              float flabels[num_total_labels] =
                  {1.0f, (float)xmin, (float)ymin, (float)xmax, (float)ymax, 1.0f / w, 1.0f / h, 1.0f};
              for (int j = 0; j < num_total_labels; ++j)
              {
                cv::Mat roi(*labels[j], r);
                roi = cv::Scalar(flabels[j]);
              }

              if (param.use_mask())
              {
                cv::Mat mask2(full_label_height,
                              full_label_width, CV_32F,
                                 cv::Scalar(0));
                cv::resize(mask, mask2, mask2.size());
                for (int y = 0; y < full_label_height; ++y)
                {
                  for (int x = 0; x < full_label_width; ++x)
                  {
                      if (mask2.at<float>(y,x)>=0.5f)
                          box_mask.at<float>(y,x) = (float)id;
                  }
                }
              }
              else // ֱ����box_mask���洦��
              {
                cv::rectangle(box_mask, r, cv::Scalar(id), -1);
              }
              hlhe_valid_bbox_cnt += 1;
          }
          else
          {
              picgen.clear();
          }

      }
  }

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

 // �޸Ĵ˴�,��֧��Hard ming�еĺ�������
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

  // handle catalog
  float *ptr = label_type;
  for (int y = 0; y < type_label_height; ++y)
  {
    for (int x = 0; x < type_label_width; ++x)
    {
      int id = (int)(box_mask.at<float>(y*type_stride,x*type_stride));
      if (id>=0 && id<itypes.size())
        *ptr = itypes[id]+1;
      else
      if (id == -1)
        *ptr = 0;
      else
      {
          LOG(ERROR) << "invalid id " << id << " " << y << ' ' << x << ' ' <<
                        (box_mask.at<float>(y*type_stride,x*type_stride)) << ' ' <<
                        data.car_img_source();
      }
      ptr++;
    }
  }

// fix the label edge, to get accurate label
#ifdef GEN_MASK1_MASK2

  for(int i=0; i<path1_mask.cols; i++)
  {
      for( int j=0; j<path1_mask.rows; j++)
      {
          float mask_ori_val = box_mask.at<float>(j,i);
          float mask1_val = path1_mask.at<float>(j,i);
          float mask2_val = path2_mask.at<float>(j,i);
          float mask3_val = path3_mask.at<float>(j,i);

          // ƽʱ����ͨ����,��ǰ��bbox�Ļ����Ͼ��ޱ�ע
          if( normal_status )
          {
              // fix the edge zone to be "gray zone" -hlhe
              if( mask1_val!=BACKGROUND_LAB &&
                  mask_ori_val==-1)
              {
                  path1_mask.at<float>(j,i) = BACKGROUND_LAB;
              }

              if( mask2_val!=BACKGROUND_LAB &&
                  mask_ori_val==-1)
              {
                  path2_mask.at<float>(j,i) = BACKGROUND_LAB;
              }

              if( mask3_val!=BACKGROUND_LAB &&
                  mask_ori_val==-1)
              {
                  path3_mask.at<float>(j,i) = BACKGROUND_LAB;
              }
          }
          else // �������ھ�,��Ե������������
          {
              // mask1
              if( IGNORE_EDGE_LAB==mask1_val )      //��Ҫ����������1
              {
                  if( mask_ori_val!=-1 )
                      path1_mask.at<float>(j,i) = TRAFFIC_SIGN_LAB;
                  else
                      path1_mask.at<float>(j,i) = GRAY_ZONE_LAB;
              }
              else if( TRAFFIC_SIGN_LAB==mask1_val ) //��Ҫ����������2
              {
                  if( -1==mask_ori_val )
                      path1_mask.at<float>(j,i) = BACKGROUND_LAB;

              }

              // mask2
              if( IGNORE_EDGE_LAB==mask2_val )
              {
                  if( mask_ori_val!=-1 )
                      path2_mask.at<float>(j,i) = TRAFFIC_SIGN_LAB;
                  else
                      path2_mask.at<float>(j,i) = GRAY_ZONE_LAB;
              }
              else if( TRAFFIC_SIGN_LAB==mask2_val )
              {
                  if( -1==mask_ori_val )
                      path2_mask.at<float>(j,i) = BACKGROUND_LAB;
              }

              // mask3
              if( IGNORE_EDGE_LAB==mask3_val )
              {
                  if( mask_ori_val!=-1 )
                      path3_mask.at<float>(j,i) = TRAFFIC_SIGN_LAB;
                  else
                      path3_mask.at<float>(j,i) = GRAY_ZONE_LAB;
              }
              else if( TRAFFIC_SIGN_LAB==mask3_val )
              {
                  if( -1==mask_ori_val )
                      path3_mask.at<float>(j,i) = BACKGROUND_LAB;
              }
          }
      }
  }

  assert(my_mask_ptr.size()==1);
  Dtype* mask_ptr = my_mask_ptr[0];
  int ncols = path1_mask.cols;
  int nrows = path1_mask.rows;

  for(int mask_idx=0; mask_idx<MASK_NUM; mask_idx++)
  {
      for(int j=0; j<nrows; j++)
      {
          for(int i=0; i<ncols; i++)
          {
              Dtype val = 0.f;
              switch(mask_idx)
              {
              case 0:
                  val = (Dtype)path1_mask.at<float>(j,i);
                  break;
              case 1:
                  val = (Dtype)path2_mask.at<float>(j,i);
                  break;
              case 2:
                  val = (Dtype)path3_mask.at<float>(j,i);
                  break;
              default:
                  assert(0);
              }

              int offset = mask_idx*ncols*nrows + j*ncols + i;
              mask_ptr[offset] = val;
          }
      }
  }


#endif


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
  int mask_one_batch_count = 0;

  if (this->output_labels_)           // generate labels?? -hlhe
  {
    Dtype *top_label = batch->label().mutable_cpu_data();
    top_labels.push_back(top_label);
    label_mask = batch->label(1).mutable_cpu_data();
    mask_one_batch_count = batch->label(1).shape()[1] * batch->label(1).shape()[2] * batch->label(1).shape()[3];
    label_type = batch->label(2).mutable_cpu_data();  // commented by hlhe
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

    vector<vector<int> > bb_pts;                       // ����������bbox����
    vector<Dtype*> mask_muta_data;
    int batch_pos = item_id * mask_one_batch_count;
    mask_muta_data.push_back(label_mask+batch_pos); // output pointers

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
                                              is_normal);

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

    // ���ɷ����õı�ע
    vector<Batch<Dtype>*> batch_vec;
    batch_vec.push_back(batch);

    lab_auxi_.gen_gray_zone(bb_pts);
    lab_auxi_.apply_gray_zone(label_datums[0], 0, Lab4ClsChanelIdx,
                              batch_vec, item_id);

    if (bid+1 >= crop_num*data.car_boxes_size())
        need_new_data = true;

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
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
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
{\
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
