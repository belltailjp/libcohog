#include "gtest/gtest.h"
#include "CoHOGDetectorCUDA.hpp"

#include <opencv2/opencv.hpp>


class TestCoHOGDetectorCUDA : public ::testing::Test
{
protected:

    libcohog::gpu_context context;

    virtual void SetUp()
    { }
};

TEST_F(TestCoHOGDetectorCUDA, transfer_image)
{
    cv::Mat_<unsigned char> img = cv::Mat_<unsigned char>::eye(1024, 768);
    libcohog::set_image(context, img.data, img.cols, img.rows);

    for(int y = 0; y < img.rows; ++y)
        for(int x = 0; x < img.cols; ++x)
            EXPECT_EQ(img(y, x), img.data[y * img.cols + x]);

    //check
    EXPECT_EQ(img.cols, context.w);
    EXPECT_EQ(img.rows, context.h);

    //check image
    context.download();
    for(int i = 0; i < context.w * context.h; ++i)
        EXPECT_EQ(img.data[i], context.img_cpu[i]);
}

TEST_F(TestCoHOGDetectorCUDA, calc_gradient)
{
    libcohog::calc_gradient(context, 8, 10);
}

