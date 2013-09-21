#pragma once

#include <opencv2/opencv.hpp>
#include <libcohog.hpp>

namespace libcohog
{

// An element of ground truth
struct TruthRect
{
    cv::Rect rect;
    bool     confident;
};

// Describes the evaluation statistics of 1 or more frames
struct EvaluationResult
{
    // The total number of TPs, FPs, FNs with grouping of detection results
    int nTP, nFP, nFN;

    // The total number of FPs without grouping of detection results (for calculation of FPPW)
    int nFPW;

    // The total number of detection windows, and frames.
    long long int nWnd;
    int           nImg;

    EvaluationResult()
        :nTP(0), nFP(0), nFN(0), nFPW(0), nWnd(0), nImg(0)
    { }

    EvaluationResult operator+(const EvaluationResult& eval) const;
    EvaluationResult& operator+=(const EvaluationResult& eval);
    std::string to_string(bool one_line = true) const;
    double F_value() const;
    double FPPF() const;
    double FPPW() const;
    double Recall() const;
    double Precision() const;
    double Missrate() const;
    double FP_rate() const;
};


struct VerificationResult
{
    // The number of detection windows before thresholding
    int total_windows;

    // Detection windows thresholded without grouping
    std::vector<cv::Rect>   windows;
    std::vector<bool>       TP_flags;

    // Detection windows thresholded with grouping
    std::vector<cv::Rect>   windows_grouped;
    std::vector<bool>       TP_grouped_flags;

    // The rectangles of the ground truth
    std::vector<TruthRect>  ground_truth;
    std::vector<bool>       found_flags;

    EvaluationResult        to_eval() const;
};

class PositionHeightFilter
{
    float bottom_a, bottom_b;
    float top_a, top_b;

public:
    PositionHeightFilter(float _bottom_a, float _bottom_b, float _top_a, float _top_b)
        :bottom_a(_bottom_a), bottom_b(_bottom_b), top_a(_top_a), top_b(_top_b)
    { }

    bool operator()(const cv::Rect& r)
    {
        return  r.height <= (top_a * (r.y + r.height) + top_b) &&
                (bottom_a * (r.y + r.height) + bottom_b) <= r.height;
    }
};


std::vector<cv::Rect>   thresholding(const std::vector<Window>& windows, double th);
std::vector<cv::Rect>   grouping(const std::vector<cv::Rect>& rects, int groupTh, double eps);
cv::Rect                normalize_rectangle(const cv::Rect& r, double height_to_width_ratio, double height_ratio);
bool                    is_equivalent(const cv::Rect& r1, const cv::Rect& r2, double overwrap_th);

template<class TFilter>
VerificationResult      verify(const DetectionResult& detection, const std::vector<TruthRect>& truth,
                                    double threshold, double overwrap_th, int groupTh, double eps, TFilter filter)
{
    libcohog::VerificationResult result;

    result.total_windows    = detection.window_cnt;
    result.windows          = libcohog::thresholding(detection.windows, threshold);
    result.windows_grouped  = libcohog::grouping(result.windows, groupTh, eps);
    result.ground_truth     = truth;

    result.TP_flags         = std::vector<bool>(result.windows.size(),          false);
    result.TP_grouped_flags = std::vector<bool>(result.windows_grouped.size(),  false);
    result.found_flags      = std::vector<bool>(result.ground_truth.size(),     false);

    // Evaluation of non-grouped detection windows
    for(int i = 0; i < result.windows.size(); ++i)
    {
        const cv::Rect& r = result.windows[i];
        if(!filter(r))
            continue;
        for(int j = 0; j < result.ground_truth.size(); ++j)
        {
            if(libcohog::is_equivalent(result.ground_truth[j].rect, r, overwrap_th))
            {
                result.TP_flags[i] = true;
                break;
            }
        }
    }

    // Evaluation of grouped detection windows
    for(int i = 0; i < result.windows_grouped.size(); ++i)
    {
        const cv::Rect& r = result.windows_grouped[i];
        if(!filter(r))
            continue;
        for(int j = 0; j < result.ground_truth.size(); ++j)
        {
            if(!result.found_flags[j] && libcohog::is_equivalent(libcohog::normalize_rectangle(result.ground_truth[j].rect, 2, 0.8), r, overwrap_th))
            {
                // Mark this rectangle in the ground truth as detected correctly.
                result.found_flags[j] = true;

                // Mark this rectangle in the detection result as true positive
                result.TP_grouped_flags[i] = true;

                break;
            }
        }
    }

    return result;
}

template<class TFilter>
EvaluationResult        evaluate(const DetectionResult& detection, const std::vector<TruthRect>& truth, 
                            double threshold, double overwrap_th, int groupTh, double eps, TFilter filter)
{
    const VerificationResult result = libcohog::verify(detection, truth, threshold, overwrap_th, groupTh, eps, filter);
    return result.to_eval();
}

VerificationResult      verify(const DetectionResult& detection, const std::vector<TruthRect>& truth, 
                                    double threshold, double overwrap_th, int groupTh, double eps);

EvaluationResult        evaluate(const DetectionResult& detection, const std::vector<TruthRect>& truth, 
                            double threshold, double overwrap_th, int groupTh, double eps);
}

