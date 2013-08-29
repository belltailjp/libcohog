#include "Detector.hpp"

namespace libcohog
{
    
std::vector<Window> Detector::detect_multi_scale(const cv::Mat_<unsigned char>& img)
{
    std::vector<Window> results;

    for(float scale = param_scan.MinScale; scale <= param_scan.MaxScale; scale *= param_scan.ScaleFactor)
    {
        const int w = static_cast<int>(img.cols / scale);
        const int h = static_cast<int>(img.rows / scale);
        cv::Mat_<unsigned char> img_scaled;
        cv::resize(img, img_scaled, cv::Size(w, h));

        const std::vector<Window> result = detect(img_scaled);
        for(int i = 0; i < result.size(); ++i)
        {
            Window r = result[i];
            r.x *= scale;
            r.y *= scale;
            r.w *= scale;
            r.h *= scale;
            results.push_back(r);
        }
    }

    std::sort(results.rbegin(), results.rend());  //sort by detection score descending

    return results;
}

}
