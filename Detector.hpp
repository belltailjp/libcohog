#pragma once

#include <iostream>
#include <cstdio>
#include <vector>
#include <opencv2/opencv.hpp>
#include <libcohog/CoHOGParams.hpp>

namespace libcohog
{

struct Window
{
    int x, y, w, h;
    double v;

    int operator<(const Window& t) const
    {
        return v < t.v;
    }

    std::string to_string() const
    {
        char buf[64];
        std::sprintf(buf, "%d %d %d %d %.6f", x, y, w, h, v);
        return std::string(buf);
    }
};

struct DetectionResult
{
    std::string filename;
    int window_cnt;
    std::vector<Window> windows;
};

void                         write_detection_windows(std::ostream& os,     const DetectionResult& result, float omit_rate = 0.95);

class Detector
{
protected:

    ScanParams param_scan;
    int w_window, h_window;

public:

    explicit Detector(const ScanParams& _param_scan)
        :param_scan(_param_scan)
    {
    }

    virtual std::vector<Window> detect(const cv::Mat_<unsigned char>& img) = 0;
    std::vector<Window> detect_multi_scale(const cv::Mat_<unsigned char>& img);
    virtual std::vector<float> calculate_feature(const cv::Mat_<unsigned char>& img) = 0;

    int get_window_width()  const { return w_window; }
    int get_window_height() const { return h_window; }
};

}

