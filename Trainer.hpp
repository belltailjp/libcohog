#pragma once

#include <opencv2/opencv.hpp>
#include <libcohog/liblinear/linear.h>

namespace libcohog
{

parameter default_liblinear_parameter();

model* train_liblinear(const std::vector<std::vector<float> >& positive_features, const std::vector<std::vector<float> >& negative_features, parameter param = default_liblinear_parameter());
model* train_liblinear(const std::vector<std::vector<feature_node> >& positive_features, const std::vector<std::vector<feature_node> >& negative_features, int dim, parameter param = default_liblinear_parameter());

}

