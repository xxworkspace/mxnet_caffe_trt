### MXNet2Caffe2trt: Convert MXNet model to Caffe model,Then Convert Caffe model to TRT model

  * first:Convert Mxnet model to Caffe model
  * then :Convert Caffe model to Tensorrt model

## Mxnet2Caffe

 * `json2prototxt.py  prototxt_basic.py` Read mxnet_json file and converte to prototxt.
 * `mxnet2caffe.py` Read mxnet_model params_dict and converte to .caffemodel.
 * `mxnet_caffe_model_test.py` Compare the outputs of Caffe model and TRT model.
 * `mxnet_t2t.py` Convert training model to inference model.Note you should change the file for your model.
 * Usage
   * `First` : Using json2prototxt.py to convert json file to prototxt. Using json2prototxt.py -h to get the args
   * `Then`  : Using mxnet2caffe.py to convert params file to caffemodel. Using mxnet2caffe.py -h to get the args
   * `Final` : Using mxnet_caffe_model_test.py to compare the model outputs. Using mxnet_caffe_model_test.py -h to get the args
   * Note:Uisng Netron(https://github.com/lutzroeder/Netron) to see the model structure

## caffe2trt_int8

 * `caffe2trt_int8.h` Convert Caffe model to TRT Int8 model.
 * `calibrator.h` Get the Calibratir for int8 model.
 * `CopyPlugin.h` Example for Add Plugin.
 * Note
   * From https://blog.csdn.net/qq_32043199/article/details/81119357, you can get the detail about float32 to int8
   * In toint8.cpp, you can see how to use these code to convert the Caffe model to TRT Int8 model

## caffe2trt

 * `caffe2trt_half.h` Convert Caffe model to TRT model. Note half mode will be used if device supports(NVIDIA Turing GPU architecture)

## trt_inference

 * `trt2engine.h` Load TRT model and do model Inference. Note just one input and one output are supported now.

## Build
 
 * docker pull sunny820828449/mxnet2trt
 * docker run -t -i sunny820828449/mxnet2trt
 * git clone http://git.sysop.bigo.sg/feature-db/mxnet_caffe_trt.git
 * Enter mxnet_caffe_trt and bash mk.sh

 
