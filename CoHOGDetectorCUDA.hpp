#pragma once
//only included locally

#include <libcohog/CoHOGParams.hpp>

namespace libcohog
{

struct gpu_context
{
    unsigned char *img;
    unsigned char *grad;

    int w, h;

    gpu_context()
        :img(NULL), grad(NULL), w(0), h(0)
    {
    }

    unsigned char *img_cpu;
    unsigned char *grad_cpu;

    void download();
};

void set_image(gpu_context& context, const unsigned char* ptr, int w, int h);
void calc_gradient(libcohog::gpu_context& context, int level, float th);

}

