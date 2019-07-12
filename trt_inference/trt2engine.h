#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include "logger.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;

#define CHECK(status) \
    if (status != 0) std::cout << "Status Error!" << std::endl;

struct InferenceEngine {
  int inputIndex;
  int outputIndex;

  void* buffers[2];
  cudaStream_t stream;
  size_t INPUT_BLOB_SIZE;
  size_t OUTPUT_BLOB_SIZE;

  IRuntime* runtime;
  ICudaEngine* engine;
  IExecutionContext* context;
  InferenceEngine() :runtime(NULL), engine(NULL), context(NULL) {}
  ~InferenceEngine() {}

  void init(size_t gpuid,std::string trt_file,
    std::string INPUT_BLOB_NAME,std::string OUTPUT_BLOB_NAME) {
    CHECK(cudaSetDevice(gpuid));
    CHECK(cudaStreamCreate(&stream));

    trt2context(trt_file);
    context_init(INPUT_BLOB_NAME,OUTPUT_BLOB_NAME);
  }

  void release() {
    CHECK(cudaFree(buffers[0]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaStreamDestroy(stream));
    runtime->destroy();
    engine->destroy();
    context->destroy();
  }

  size_t get_input_blob_size(){
    return INPUT_BLOB_SIZE;
  }
  size_t get_output_blob_size(){
    return OUTPUT_BLOB_SIZE;
  }

  void trt2context(std::string trt_file) {
    std::ifstream ifs(trt_file, std::ios::binary);
    if (ifs.fail()) {
      std::cout << "file open fail!" << std::endl;
      exit(0);
    }
    ifs.seekg(0,std::ios::end);
    uint64_t size = ifs.tellg();
    char* serial = (char*)malloc(size);
    ifs.seekg(0,std::ios::beg);

    ifs.read(serial, size);
    ifs.close();

    Logger gLogger;
    runtime = createInferRuntime(gLogger.getTRTLogger());
    engine = runtime->deserializeCudaEngine(serial, size, NULL);
    context = engine->createExecutionContext();
  }

  void context_init(
    std::string INPUT_BLOB_NAME,std::string OUTPUT_BLOB_NAME)
  {
    const ICudaEngine& engine = context->getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    if (engine.getNbBindings() != 2) exit(0);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME.c_str());
    outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME.c_str());

    DimsCHW dim_i = static_cast<DimsCHW&&>(context->getEngine().getBindingDimensions(inputIndex));
    DimsCHW dim_o = static_cast<DimsCHW&&>(context->getEngine().getBindingDimensions(outputIndex));

    size_t batch_size = engine.getMaxBatchSize();
    INPUT_BLOB_SIZE = dim_i.c()*dim_i.h()*dim_i.w() * sizeof(float);
    OUTPUT_BLOB_SIZE = dim_o.c()*dim_o.h()*dim_o.w() * sizeof(float);
    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], INPUT_BLOB_SIZE*batch_size));
    CHECK(cudaMalloc(&buffers[outputIndex], OUTPUT_BLOB_SIZE*batch_size));
  }

  void doInference(int batchsize,void*input,void*output) {
    cudaMemcpyAsync(buffers[inputIndex], input, batchsize * INPUT_BLOB_SIZE, cudaMemcpyHostToDevice, stream);
    context->enqueue(batchsize, buffers, stream, nullptr);
    cudaMemcpyAsync(output, buffers[outputIndex], batchsize * OUTPUT_BLOB_SIZE, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }
};
