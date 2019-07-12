
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <NvInfer.h>
#include "trt2engine.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace nvinfer1;

void loadImage(std::string& path,
               float* data,
               int h,int w,
               std::vector<float>& mean,
               std::vector<float>& scale) {
  cv::Mat img = cv::imread(path);

  cv::Mat dst;
  cv::resize(img, dst, cv::Size(w, h));

  float* r, *g, *b;
  r = data;
  g = data + w * h;
  b = data + w * h * 2;
  uchar* tmp = dst.data;
  for (int i = 0; i < h * w; ++i) {
    *(b++) = (static_cast<float>(*(tmp++)) - mean[2]) * scale[2];
    *(g++) = (static_cast<float>(*(tmp++)) - mean[1]) * scale[1];
    *(r++) = (static_cast<float>(*(tmp++)) - mean[0]) * scale[0];
  }
}

void getImageList(std::string m_CalibratorImagePath,std::vector<std::string>& m_ImageList,std::vector<std::string>& labels) {
  std::ifstream ifs(m_CalibratorImagePath, std::ios::in);
  if (ifs.fail()) exit(0);

  std::string line;
  while (true) {
    getline(ifs, line);
    if (line.size() <= 1)
      break;

    std::cout << "loading : " << line << std::endl;
    stringstream sstr(line);
    std::string tmp;
    sstr >> tmp;
    m_ImageList.push_back(tmp);
    sstr >> tmp;
    labels.push_back(tmp);
    //m_ImageList.push_back(line);
  }
  ifs.close();
}

void getBatch(int& batch,
              float* data,
              std::vector<std::string>& m_ImageList,
              int index,
              int c,int h,int w,
              std::vector<float>& mean,
              std::vector<float>& scale) {
  batch = batch < m_ImageList.size() - index ? batch : m_ImageList.size() - index;

#pragma omp parallel  for
  for (int i = 0; i < batch; ++i)
    loadImage(m_ImageList[index++], data + i * (c * h * 3), h, w, mean, scale);
}

template<class T>
T t2t(std::string str){
  std::stringstream sstr(str);
  T tmp;
  sstr >> tmp;
  return tmp;
}

int main(int argc,char**argv){
  if(argc != 9 && argc != 12 && argc != 15){
    std::cout<<"trt_model    #trt model path"<<std::endl
             <<"image_file   #test image path"<<std::endl
             <<"gpu_id       #gpu id"<<std::endl
             <<"batch_size   #batch size"<<std::endl
             <<"model_otuput #model output name"<<std::endl
             << "c,h,w       #model input size"<<std::endl
             << "r_mean,g_mean,b_mean     #image mean"<<std::endl
             << "r_scale,g_scale,b_scale  #image scale"<<std::endl;
    exit(0);
  }
  std::string trt_model  = std::string(argv[1]);
  std::string image_file = std::string(argv[2]);
  
  std::string outputs(argv[5]);
  int32_t gpuid = t2t<int>(argv[3]);
  int32_t batch = t2t<int>(argv[4]);

  int32_t c = t2t<int>(argv[6]);
  int32_t h = t2t<int>(argv[7]);
  int32_t w = t2t<int>(argv[8]);

  std::vector<float> mean = { 123.0, 117.0, 104.0 }, scale = {1.0 , 1.0 , 1.0};
  if(argc >= 12){
    mean[0] = t2t<float>(argv[9]);
    mean[1] = t2t<float>(argv[10]);
    mean[2] = t2t<float>(argv[11]);
  }
  if(argc == 15){
    scale[0] = t2t<float>(argv[12]);
    scale[1] = t2t<float>(argv[13]);
    scale[2] = t2t<float>(argv[14]);
  }
  InferenceEngine infer;
  infer.init(0, trt_model, "data", outputs);

  float* data = (float*)malloc(infer.get_input_blob_size() * batch);
  float* output = (float*)malloc(infer.get_output_blob_size() * batch);

  std::vector<std::string> images, labels;
  getImageList(image_file,images,labels);

  float accuracy = 0;
  for (int i = 0; i < images.size(); i += batch) {
    getBatch(batch, data, images, i, c, h, w, mean, scale);
    infer.doInference(batch, data, output);
    for (int j = 0; j < batch ; ++j) {
      if (output[j * 2] > 0.5 && labels[i + j] == "0.0")
        accuracy += 1;

      if (output[j * 2] < 0.5 && labels[i + j] == "1.0")
        accuracy += 1;
    }
  }
  std::cout <<"accuracy : "<< accuracy / images.size() << std::endl;

  return 0;
}

