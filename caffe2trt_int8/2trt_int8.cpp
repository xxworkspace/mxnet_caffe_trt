
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "caffe2trt_int8.h"

//#pragma commit(lib,"nvinfer.lib")
//#pragma commit(lib,"nvparsers.lib")
//#pragma commit(lib,"opencv_world346d.lib")

void getOutput(std::vector<std::string>& outputs, std::string filename) {
  std::ifstream ifs(filename);
  std::string line;
  while (true) {
    getline(ifs,line);
    if (line.size() <= 1)
      break;
    outputs.push_back(line);
  }
  ifs.close();
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
    std::cout<<"caffe-model-name-prefix  #caffe model prefix path"<<std::endl
             <<"calibrator-image-file    #calibrator image file path"<<std::endl
             <<"gpu_id                   #gpu id"<<std::endl
             <<"batch_size               #batch size"<<std::endl
             <<"model_otuput             #model output name"<<std::endl
             << "c,h,w                   #model input size"<<std::endl
             << "r_mean,g_mean,b_mean    #image mean"<<std::endl
             << "r_scale,g_scale,b_scale #image scale"<<std::endl;
    exit(0);
  }
  std::string prefix = std::string(argv[1]);  //"D:\\workspace\\mxnet_caffe_tensorrt\\Mxnet2Caffe-Tensor-RT-SEnet\\caffemodels\\";
  std::string deploy = prefix + ".prototxt";
  std::string model  = prefix + ".caffemodel";
  std::string calibrator_int8 = prefix + ".calibrator";
  std::string trt_model       = prefix + ".trt";
  std::string calibrator_image_file = argv[2];
  
  std::vector<std::string> outputs;
  outputs.push_back(argv[5]);

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

  IInt8Calibrator *calibrator = new Calibrator(
    gpuid, batch, calibrator_int8, "", calibrator_image_file, c, h, w, "data", mean, scale);
  caffeToTRTModel(deploy, model, outputs, batch, calibrator, trt_model);

  return 0;
}

