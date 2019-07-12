#pragma once

#include "logger.h"
#include <NvCaffeParser.h>

#include <vector>
#include <string>
#include <fstream>
#include "calibrator.h"
#include "CopyPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

void caffeToTRTModel(
	const std::string& deployFile,                    // name for caffe prototxt
	const std::string& modelFile,                     // name for caffemodel 
	const std::vector<std::string>& outputs,          // network outputs
	unsigned int maxBatchSize,                        // batch size - NB must be at least as large as the batch we want to run with)
	IInt8Calibrator *calibrator,
	std::string serial_file)                          // output stream for the TensorRT model
{
	Logger gLogger;
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	//parser->setPluginFactoryV2((nvcaffeparser1::IPluginFactoryV2*)new PluginFactory());
	const IBlobNameToTensor* blobNameToTensor = parser->parse(
		deployFile.c_str(), modelFile.c_str(),
		*network, DataType::kFLOAT);
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setInt8Calibrator(calibrator);
	builder->setInt8Mode(true);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	// serialize the engine, then close everything down
	IHostMemory* model{ nullptr };
	model = engine->serialize();

	std::ofstream ofs(serial_file, std::ios::binary);
	if (ofs.fail()) {
		std::cerr << "serial file open fail!" << std::endl;
		exit(0);
	}

	ofs.write((char*)model->data(), model->size());
	ofs.close();

	network->destroy();
	parser->destroy();

	model->destroy();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}
