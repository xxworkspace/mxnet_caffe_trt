#pragma once

#include "logger.h"
#include <NvCaffeParser.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <string>
#include <fstream>

using namespace nvinfer1;
using namespace nvcaffeparser1;

void caffeToTRTModel(
  const std::string& deployFile,                    // name for caffe prototxt
  const std::string& modelFile,                     // name for caffemodel 
  const std::vector<std::string>& outputs,          // network outputs
  int gpuid,                                        // gpu_id
  int maxBatchSize,                                 // batch size - NB must be at least as large as the batch we want to run with)
  bool  datatype,                                   // datatype
  std::string serial_file)                          // output stream for the TensorRT model
{
  Logger gLogger;
  // create the builder
  cudaSetDevice(gpuid);
  IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
  // parse the caffe model to populate the network, then set the outputs
  INetworkDefinition* network = builder->createNetwork();
  ICaffeParser* parser = createCaffeParser();

  bool fp16 = datatype ? builder->platformHasFastFp16() : false;
  const IBlobNameToTensor* blobNameToTensor = parser->parse(
    deployFile.c_str(), modelFile.c_str(),
    *network, fp16 ? DataType::kHALF : DataType::kFLOAT);

  // specify which tensors are outputs
  for (auto& s : outputs)
    network->markOutput(*blobNameToTensor->find(s.c_str()));

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 30);
  builder->setFp16Mode(fp16);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  network->destroy();
  parser->destroy();

  // serialize the engine, then close everything down
  IHostMemory* model{ nullptr };
  model = engine->serialize();

  std::ofstream ofs(serial_file, std::ios::binary);
  if (ofs.fail()) {
    std::cerr << "serial file open fail!" << std::endl;
    exit(0);
  }

  ofs.write((char*)model->data(),model->size());
  ofs.close();

  model->destroy();
  engine->destroy();
  builder->destroy();
  shutdownProtobufLibrary();
}