#pragma once

#include <opencv2/opencv.hpp>
#include <libcohog.hpp>
#include <linear.h>

namespace libcohog
{

/*
 * スレッドセーフじゃないよ！
 */
class CoHOGDetector2: public Detector
{
    CoHOGParams param_cohog;
    int width, height;

    std::vector<double> weights;

    void initialize();
    int quantitize_gradient(int level, float th, int dx, int dy) const;
    cv::Mat_<unsigned char> calc_gradient_orientation_matrix(const cv::Mat_<unsigned char>& image, unsigned level, float th) const;
    cv::Mat_<unsigned char> calc_cooccurrence_matrix(const cv::Mat_<unsigned char>& gradient, int dx, int dy) const;
    std::vector<cv::Mat_<unsigned char> > calc_cooccurrence_matrices(const cv::Mat_<unsigned char>& gradient) const;

    void calc_feature(const cv::Mat_<unsigned char>& img, std::vector<unsigned char>& feature, int bx, int by, int w, int h) const;

public:

    CoHOGDetector2(int image_width, int image_height, const CoHOGParams& _param_cohog = CoHOGParams(), const ScanParams& _param_scan = ScanParams())
        :width(image_width), height(image_height),
         Detector(_param_scan),
         param_cohog(_param_cohog),
         weights(_param_cohog.dimension(), 0)
    {
        w_window = param_cohog.width();
        h_window = param_cohog.height();
        dim      = param_cohog.dimension();

        initialize();
    }

    void set_detector(const std::vector<double>& _weights);
    void set_detector(model *liblinear_model);
    void set_detector(const char* liblinear_model_file);

    std::vector<Window> detect(const cv::Mat_<unsigned char>& img);
    std::vector<float> calculate_feature(const cv::Mat_<unsigned char>& img){}
};

}

