#include <libcohog/HOGDetector.hpp> 

namespace libcohog
{

HOGDetector::HOGDetector(const ScanParams& _param_scan)
    :Detector(_param_scan)
{
    //descriptor.setSVMDetector();

#ifdef WITH_CUDA
    w_window = descriptor.win_size.width;
    h_window = descriptor.win_size.height;
#else
    w_window = descriptor.winSize.width;
    h_window = descriptor.winSize.height;
#endif

}


std::vector<Window> HOGDetector::detect(const cv::Mat_<unsigned char>& img)
{
    std::vector<cv::Point> found;       // dummy
    std::vector<cv::Point> locations;
    std::vector<double> confidences;

#ifdef WITH_CUDA

    cv::gpu::GpuMat img_gpu;
    img_gpu.upload(img);

    descriptor.computeConfidence(img_gpu, found, 0, cv::Size(8,8), cv::Size(0,0), locations, confidences);

#else

    descriptor.detect(img, found, confidences, 0, cv::Size(8, 8), cv::Size(0, 0), locations);

#endif

    // convert to detection window description form
    std::vector<Window> result;
    for(int idx = 0; idx < locations.size(); ++idx)
    {
        Window w;
        w.v = confidences[idx];
        w.x = locations[idx].x;
        w.y = locations[idx].y;
        w.w = w_window;
        w.h = h_window;
        result.push_back(w);
    }
    std::sort(result.rbegin(), result.rend());  //sort by detection score descending

    return result;
}


std::vector<float> HOGDetector::calculate_feature(const cv::Mat_<unsigned char>& img)
{
    std::vector<float> feature;

#ifdef WITH_CUDA

    cv::gpu::GpuMat img_gpu;    // images uploaded to GPU
    cv::gpu::GpuMat desc_gpu;   // feature caluclated on GPU
    cv::Mat_<float> desc;       // feature downloaded from GPU

    img_gpu.upload(img);
    descriptor.getDescriptors(img_gpu, cv::Size(8, 8), desc_gpu);
    desc_gpu.download(desc);

    for(int col = 0; col < desc.cols; ++col)
        feature.push_back(desc(0, col));

#else
    descriptor.compute(img, feature);
#endif

    return feature;
}

}

