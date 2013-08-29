#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <libcohog/CoHOGParams.hpp>

namespace libcohog
{

struct Window
{
    int x, y, w, h;
    float v;

    int operator<(const Window& t) const
    {
        return v < t.v;
    }
};


class Detector
{
protected:

    ScanParams param_scan;

public:

    explicit Detector(const ScanParams& _param_scan)
        :param_scan(_param_scan)
    {
    }

    virtual std::vector<Window> detect(const cv::Mat_<unsigned char>& img) = 0;
    std::vector<Window> detect_multi_scale(const cv::Mat_<unsigned char>& img);
    virtual std::vector<float> calculate_feature(const cv::Mat_<unsigned char>& img) = 0;
};

}
