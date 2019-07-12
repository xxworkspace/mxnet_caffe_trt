
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

#include <NvInfer.h>
#include <NvCaffeParser.h>
#include "caffe2trt_half.h"

using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;

template<class T>
T t2t(std::string str){
  std::stringstream sstr(str);
  T tmp;
  sstr >> tmp;
  return tmp;
}
int main(int argc,char**argv){
  if(argc != 5){
    std::cout<<"caffe-model-name-prefix  #caffe model prefix path"<<std::endl
             <<"gpu_id                   #gpu id"<<std::endl
             <<"batch_size               #batch size"<<std::endl
             <<"model_otuput             #model output name"<<std::endl;
    exit(0);
  }
  std::string prefix = std::string(argv[1]);  //"D:\\workspace\\mxnet_caffe_tensorrt\\Mxnet2Caffe-Tensor-RT-SEnet\\caffemodels\\";
  std::string deploy = prefix + ".prototxt";
  std::string model  = prefix + ".caffemodel";
  std::string trt_model       = prefix + ".trt";
  
  std::vector<std::string> outputs;
  outputs.push_back(argv[4]);

  int32_t gpuid = t2t<int>(argv[2]);
  int32_t batch = t2t<int>(argv[3]);
  caffeToTRTModel(deploy, model, outputs, gpuid, batch, true, trt_model);
  
  return 0;
}
