#pragma once

#include <string.h>
#include <memory>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace nvcaffeparser1;

class CopyPlugin : public IPluginV2
{
public:
  CopyPlugin(){}
  // create the plugin at runtime from a byte stream
  CopyPlugin(const void* data, size_t length){
    const char *d = static_cast<const char*>(data), *a = d;
    read(d, mDataType);
    read(d, mDataSize);
  }
  CopyPlugin(const CopyPlugin& plugin):
    mDataType(plugin.mDataType),
    mDataSize(plugin.mDataSize){}

  ~CopyPlugin(){}
  virtual const char* getPluginType()const {
    return "Copy";
  } 
  virtual const char* getPluginVersion()const {
    return "IPluginV2_Copy";
  }

  int getNbOutputs() const override{
    return 1;
  }

  Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override{
    return Dims3(inputs->d[0], inputs->d[1], inputs->d[2]);
  }

  bool supportsFormat(DataType type, PluginFormat format) const override { return true; }

  void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override{
    mDataType = type;
    mDataSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
    //std::cout << mDataSize << std::endl;
  }

  int initialize() override{
    return 0;
  }

  virtual void terminate() override{}

  virtual size_t getWorkspaceSize(int maxBatchSize) const override{
    return 0;
  }

  virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override{
    //std::cout << mDataSize << std::endl;
    if (mDataType == DataType::kFLOAT) {
      cudaMemcpyAsync(outputs[0], inputs[0], sizeof(float)*mDataSize,cudaMemcpyDeviceToDevice, stream);
    }
    else  if (mDataType == DataType::kHALF) {
      cudaMemcpyAsync(outputs[0], inputs[0], sizeof(short)*mDataSize, cudaMemcpyDeviceToDevice, stream);
    }
    else {
      cudaMemcpyAsync(outputs[0], inputs[0], mDataSize, cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);
    return 0;
  }

  virtual size_t getSerializationSize()const {
    return sizeof(mDataSize) + sizeof(mDataType);
  }

  virtual void serialize(void* buffer)const {
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDataType);
    write(d, mDataSize);
  }
  virtual void destroy() {}
  virtual IPluginV2* clone()const {
    return (IPluginV2*)new CopyPlugin(*this);
  }
  virtual void setPluginNamespace(const char* pluginNamespace) {}
  virtual const char* getPluginNamespace() const { return ""; }
private:
  template <typename T>
  void write(char*& buffer, const T& val)const{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
  }

  template <typename T>
  void read(const char*& buffer, T& val)const{
    val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
  }

  DataType mDataType{ DataType::kFLOAT };
  size_t mDataSize{ 0 };
};

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryV2
{
public:
  // caffe parser plugin implementation
  bool isPluginV2(const char* name) override{
    std::string pname(name);
    auto pos = pname.find("Copy");
    if (pos != std::string::npos) {
      std::cout << "Find Layer Copy!" << std::endl;
      return true;
    }
    else 
      return false;
  }

  virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights, const char* libNamespace = ""){
    mPlugin = std::unique_ptr<CopyPlugin>(new CopyPlugin());
    return mPlugin.get();
  }

  // deserialization plugin implementation
  virtual IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) {
    //This plugin object is destroyed when engine is destroyed by calling
    //IPluginExt::destroy()
    return (IPlugin*)new CopyPlugin(serialData, serialLength);
  }

  // User application destroys plugin when it is safe to do so.
  // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
  void destroyPlugin(){
    mPlugin.reset();
  }

  std::unique_ptr<CopyPlugin> mPlugin{ nullptr };
};
