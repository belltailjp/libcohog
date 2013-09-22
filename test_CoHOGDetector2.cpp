#include "gtest/gtest.h"

#include "CoHOGDetector2.hpp"

TEST(cohogdetector_test, hogehoge)
{
    //cv::Mat_<unsigned char> img = cv::imread("/muradata/vehicle/daimler/TestData/13m_16s_379669u.png", 0);
    cv::Mat_<unsigned char> img = cv::imread("0023492.jpg", 0);
    libcohog::CoHOGDetector2 cohog(img.cols, img.rows);
    for(int i = 0; i < 10; ++i)
        cohog.detect(img);
}

