#include "Detector.hpp"

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

}
