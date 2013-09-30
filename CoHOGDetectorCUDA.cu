#include "CoHOGDetectorCUDA.hpp"

void libcohog::set_image(libcohog::gpu_context& context, const unsigned char* ptr, int w, int h)
{
    if(context.w < w)
    {
        if(context.w != 0)
        {
            cudaFree(&context.img);
            cudaFree(&context.grad);
        }

        cudaMalloc(&context.img,  sizeof(unsigned char) * w * h);
        cudaMalloc(&context.grad, sizeof(unsigned char) * w * h);
        context.w = w;
        context.h = h;
    }

    cudaMemcpy(context.img, ptr, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
}

void libcohog::gpu_context::download()
{
    img_cpu  = new unsigned char[w * h];
    grad_cpu = new unsigned char[w * h];

    cudaMemcpy(img_cpu,  img,  sizeof(unsigned char) * w * h, cudaMemcpyDeviceToHost);
    cudaMemcpy(grad_cpu, grad, sizeof(unsigned char) * w * h, cudaMemcpyDeviceToHost);
}


__device__ unsigned char quantitze_gradient(int level, float th, int dx, int dy)
{
    if(dx * dx + dy * dy < th * th)
        return 0xff;

    const float rad     = atan2((float)dy, (float)dx);
    const int   deg     = (int)(rad * 180.0 / M_PI);
    const int   quant   = (int)floor(deg * level / 360.0f + 0.5f);
    const int   norm    = (quant + 2 + level) % level;
    return (unsigned char)norm;
}

__global__ void calc_gradient_kernel(const unsigned char* img, unsigned char* grad, int w, int h, int level, float th)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(w <= x || h <= y)
        return;

    if(x == 0 || y == 0 || w - 1 == x || h - 1 == y)
    {
        grad[y * w + x] = 0xff;
    }

    const int dx =  img[(y + 1) * w + (x + 1)] + img[y * w + (x + 1)] + img[(y - 1) * w + (x + 1)] -
                    img[(y + 1) * w + (x - 1)] - img[y * w + (x - 1)] - img[(y - 1) * w + (x - 1)];
    const int dy =  img[(y - 1) * w + (x - 1)] + img[(y - 1) * w + x] + img[(y - 1) * w + (x + 1)] -
                    img[(y + 1) * w + (x - 1)] - img[(y + 1) * w + x] - img[(y + 1) * w + (x + 1)];

    grad[y * w + x] = quantitze_gradient(level, th, dx, dy);
}

void libcohog::calc_gradient(libcohog::gpu_context& context, int level, float th)
{
    dim3 threads(32, 32);
    dim3 blocks((int)ceil(1.0 * context.w / threads.x), (int)ceil(context.h / threads.y));
    calc_gradient_kernel<<<blocks, threads>>>(context.img, context.grad, context.w, context.h, level, th);
}



