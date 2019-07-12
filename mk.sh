echo "mkdir bin"
mkdir bin
echo "build caffe2trt ..."
cd caffe2trt
g++ -std=c++11 2trt.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnvinfer -lnvparsers -o ../bin/2trt
cd ../
echo "build caffe2trt_int8 ..."
cd caffe2trt_int8
g++ -std=c++11 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 2trt_int8.cpp -lcudart -lopencv_core -lopencv_highgui -lopencv_imgproc -lnvcaffe_parser -lnvinfer -o ../bin/2trt_int8
cd ../
echo "build trt_inference ..."
cd trt_inference
g++ -std=c++11 trt_infer.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lnvinfer -lopencv_core -lopencv_highgui -lopencv_imgproc -o ../bin/trt_infer

