#pragma once

#ifdef WITH_CUDA
#include <opencv2/gpu/gpu.hpp>
using cv::gpu::HOGDescriptor;
using cv::gpu::GpuMat;
#else
#include <opencv2/opencv.hpp>
using cv::HOGDescriptor;
#endif

#include <libcohog/Detector.hpp>
#include <libcohog/CoHOGParams.hpp>

namespace libcohog
{

class HOGDetector: public Detector
{
    HOGDescriptor descriptor;

public:

    HOGDetector(const ScanParams& _param_scan = ScanParams());
    std::vector<Window> detect(const cv::Mat_<unsigned char>& img);

    std::vector<float> calculate_feature(const cv::Mat_<unsigned char>& img);
};

}

