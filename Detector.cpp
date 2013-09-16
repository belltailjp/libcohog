#include "Detector.hpp"

#include <fstream>
#include <stdexcept>

namespace libcohog
{
    
std::vector<Window> Detector::detect_multi_scale(const cv::Mat_<unsigned char>& img)
{
    std::vector<Window> results;

    const float min_scale = 1.0 * h_window / param_scan.MaxHeight;
    const float max_scale = 1.0 * h_window / param_scan.MinHeight;
    for(float scale = min_scale; scale <= max_scale; scale *= param_scan.ScaleFactor)
    {
        const int w = static_cast<int>(img.cols * scale);
        const int h = static_cast<int>(img.rows * scale);
        if(h <= h_window)
            break;

        cv::Mat_<unsigned char> img_scaled;
        cv::resize(img, img_scaled, cv::Size(w, h));

        const std::vector<Window> result = detect(img_scaled);
        for(int i = 0; i < result.size(); ++i)
        {
            Window r = result[i];
            r.x /= scale;
            r.y /= scale;
            r.w /= scale;
            r.h /= scale;
            results.push_back(r);
        }
    }

    std::sort(results.rbegin(), results.rend());  //sort by detection score descending

    return results;
}

std::vector<feature_node> Detector::calculate_feature_nodes(const cv::Mat_<unsigned char>& img)
{
    const std::vector<float> features = calculate_feature(img);

    std::vector<feature_node> feature_nodes;
    for(int i = 0; i < features.size(); ++i)
        if(features[i] != 0.0)
            feature_nodes.push_back(feature_node{.index = i + 1, .value = features[i]});
    feature_nodes.push_back(feature_node{.index = -1, .value = 0});
    return feature_nodes;
}



void write_detection_windows(std::ostream& os, const DetectionResult& result, float omit_rate)
{
    const int data_cnt = static_cast<int>(result.windows.size() * omit_rate);
    os << result.filename << " " << data_cnt << " " << result.windows.size() << std::endl;

    for(int j = 0; j < data_cnt; ++j)
        os << result.windows[j].to_string() << std::endl;
}


std::vector<DetectionResult> load_detection_windows(const char* filename)
{
    std::vector<DetectionResult> results;

    std::ifstream ifs(filename);
    if(!ifs.is_open())
        return results;

    int cnt;
    ifs >> cnt;

    for(int i = 0; i < cnt; ++i)
    {
        int data_cnt;

        DetectionResult result;
        ifs >> result.filename >> data_cnt >> result.window_cnt;

        for(int j = 0; j < data_cnt; ++j)
        {
            libcohog::Window r;
            ifs >> r.x >> r.y >> r.w >> r.h >> r.v;
            result.windows.push_back(r);
        }

        results.push_back(result);
    }
    return results;
}

}
