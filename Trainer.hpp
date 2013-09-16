#pragma once

#include <opencv2/opencv.hpp>
#include <libcohog/liblinear/linear.h>

namespace libcohog
{

static parameter default_liblinear_parameter()
{
    parameter par;
    par.solver_type  = L2R_L2LOSS_SVC_DUAL;
    par.eps          = 0.1;
    par.C            = 1;
    par.nr_weight    = 0;
    par.weight_label = NULL;
    par.weight       = NULL;
    return par;
}

model* train_liblinear(const std::vector<std::vector<float> >& positive_features, const std::vector<std::vector<float> >& negative_features, parameter param = default_liblinear_parameter());

}

