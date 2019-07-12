#pragma once

#include <omp.h>
#include <string>
#include <vector>
#include <random>
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(status) \
  if(status != cudaSuccess) std::cout<<__LINE__<<" : "<<cudaGetErrorString(status)<<std::endl;

class Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
  Calibrator(
    const uint32_t gpuId,
    const uint32_t batchSize,
    const std::string calibrationSetPath,
    const std::string calibratorImageName,
    const std::string calibratorImagePath,
    const uint64_t inputC,
    const uint32_t inputH,
    const uint32_t inputW,
    const std::string inputBlobName,
      std::vector<float> mean,
      std::vector<float> std);
    virtual ~Calibrator() { CHECK_CUDA(cudaFree(m_DeviceInput)); }

    bool getNextBatch();
    bool getImageList();

    int getBatchSize() const override { return m_BatchSize; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    const uint32_t m_GpuId;
    const uint32_t m_BatchSize;
    const uint32_t m_InputH;
    const uint32_t m_InputW;
    const uint64_t m_InputSize;
    const std::string m_InputBlobName;
    const std::string m_CalibrationSetPath{ nullptr };
    const std::string m_CalibratorImageName{ nullptr };
    const std::string m_CalibratorImagePath{ nullptr };

    std::vector<float> m_Mean;
    std::vector<float> m_Std;

    uint32_t m_ImageIndex;
    bool m_ReadCache{ true };
    float* m_Input{ nullptr };
    float* m_DeviceInput{ nullptr };
    std::vector<std::string> m_ImageList;
    std::vector<char> m_CalibrationCache;
};

Calibrator::Calibrator(
  const uint32_t gpuId,
  const uint32_t batchSize,
  const std::string calibrationSetPath,
  const std::string calibratorImageName,
  const std::string calibratorImagePath,
  const uint64_t inputC,
  const uint32_t inputH,
  const uint32_t inputW,
  const std::string inputBlobName,
  std::vector<float> mean,
  std::vector<float> std) :
    m_GpuId(gpuId),
    m_BatchSize(batchSize),
    m_InputH(inputH),
    m_InputW(inputW),
    m_InputSize(inputC*inputH*inputW),
    m_InputBlobName(inputBlobName),
    m_CalibrationSetPath(calibrationSetPath),
    m_CalibratorImageName(calibratorImageName),
    m_CalibratorImagePath(calibratorImagePath),
    m_ImageIndex(0){
    m_Mean = mean;
    m_Std = std;

    getImageList();
    std::random_shuffle(m_ImageList.begin(), m_ImageList.end(), [](int i) { return rand() % i; });
    CHECK_CUDA(cudaSetDevice(gpuId));
    m_Input = (float*)malloc(batchSize * m_InputSize * sizeof(float));
    CHECK_CUDA(cudaMalloc((void**)&m_DeviceInput, batchSize * m_InputSize * sizeof(float)));
}

bool Calibrator::getImageList() {
  std::ifstream ifs(m_CalibratorImagePath, std::ios::in);
  if (ifs.fail()) exit(0);

  std::string line;
  while (true) {
    getline(ifs,line);
    if (line.size() <= 1)
      break;
      std::cout << "loading : " << line << std::endl;
      m_ImageList.push_back(line);
    }
  ifs.close();
  return true;
}


bool Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
  //std::cout << __LINE__ << std::endl;
  if (getNextBatch()) {
    CHECK_CUDA(cudaSetDevice(m_GpuId));
    CHECK_CUDA(cudaMemcpy(m_DeviceInput, m_Input, m_BatchSize * m_InputSize * sizeof(float), cudaMemcpyHostToDevice));
    bindings[0] = m_DeviceInput;
    //strcmp(names[0], m_InputBlobName.c_str());
    return true;
  }
  else return false;
}

bool Calibrator::getNextBatch() {
  if (m_ImageIndex + m_BatchSize > m_ImageList.size()) {
    return false;
  }

#pragma omp parallel  for
  for (int i = 0; i < m_BatchSize; ++i) {
    cv::Mat img = cv::imread(m_CalibratorImageName + "/" + m_ImageList[m_ImageIndex + i]);
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(m_InputW, m_InputH));
    float* r,*g, *b;
    r = m_Input + i * m_InputSize;
    g = m_Input + i * m_InputSize + m_InputH * m_InputW;
    b = m_Input + i * m_InputSize + m_InputH * m_InputW * 2;
    auto* tmp = dst.data;
    for (int i = 0; i < m_InputH*m_InputW; ++i) {
      *(b++) = (static_cast<float>(*(tmp++)) - m_Mean[2])*m_Std[2];
      *(g++) = (static_cast<float>(*(tmp++)) - m_Mean[1])*m_Std[1];
      *(r++) = (static_cast<float>(*(tmp++)) - m_Mean[0])*m_Std[0];
    }
  }
  std::cout << "loading : " << m_ImageIndex << std::endl;
  m_ImageIndex += m_BatchSize;
  return true;
}

const void* Calibrator::readCalibrationCache(size_t& length)
{
  //std::cout << __LINE__ << std::endl;
  m_CalibrationCache.clear();
  std::ifstream ifs(m_CalibrationSetPath, std::ios::binary);
  if (ifs.fail()) {
    length = 0;
    return nullptr;
  }
  ifs.seekg(0,std::ios::end);
  size_t size = ifs.tellg();
  m_CalibrationCache.resize(size);
  ifs.seekg(0,std::ios::beg);
  ifs.read(m_CalibrationCache.data(), size);
  length = size;
  if (length) 
    return &m_CalibrationCache[0];
  else 
    return nullptr;
}

void Calibrator::writeCalibrationCache(const void* cache, size_t length)
{
  //std::cout << __LINE__ << std::endl;
  std::ofstream output(m_CalibrationSetPath, std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
  output.close();
}
