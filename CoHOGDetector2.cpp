#include "CoHOGDetector2.hpp"

#include <stdexcept>
#include <stdexcept>

namespace libcohog
{

void CoHOGDetector2::initialize()
{
}

void CoHOGDetector2::set_detector(const std::vector<double>& _weights)
{
    if(param_cohog.dimension() != _weights.size())
        throw std::invalid_argument("The dimension of given weight vector is different from the dimension of CoHOG feature");

    weights = _weights;
}

void CoHOGDetector2::set_detector(model *liblinear_model)
{
    const int dim = liblinear_model->nr_feature + 1;
    std::vector<double> weights(dim, 0);
    for(int idx = 0; idx < dim; ++idx)
        weights[idx] = liblinear_model->w[idx];
    set_detector(weights);
}

void CoHOGDetector2::set_detector(const char* liblinear_model_file)
{
    model *m = load_model(liblinear_model_file);
    set_detector(m);
    free_and_destroy_model(&m);
}


int CoHOGDetector2::quantitize_gradient(int level, float th, int dx, int dy) const
{
    if(static_cast<float>(dx * dx + dy * dy) < th * th)
        return -1;

    const double rad = std::atan2(dy, dx);
    const int deg    = static_cast<int>(rad * 180.0 / M_PI);
    const int quant  = static_cast<int>(std::floor(deg * level / 360.0 + 0.5));
    const int norm   = (quant + 2 + level) % level;
    return norm;
}

cv::Mat_<unsigned char> CoHOGDetector2::calc_gradient_orientation_matrix(const cv::Mat_<unsigned char>& image, unsigned level, float th) const
{
    cv::Mat_<unsigned char> result = cv::Mat_<unsigned char>::zeros(image.size());
    const unsigned w = image.cols;
    const unsigned h = image.rows;

    for(unsigned y = 1; y < h - 1; ++y)
    {
        for(unsigned x = 1; x < w - 1; ++x)
        {
            const int dx = image(y + 1, x + 1) + image(y, x + 1) + image(y - 1, x + 1) - image(y, x - 1) - image(y + 1, x - 1) - image(y - 1, x - 1);
            const int dy = image(y + 1, x - 1) + image(y + 1, x) + image(y + 1, x + 1) - image(y - 1, x) - image(y - 1, x + 1) - image(y - 1, x - 1);
            result(y, x) = quantitize_gradient(level, th, dx, dy);
        }
    }
    return result;
}

cv::Mat_<unsigned char> CoHOGDetector2::calc_cooccurrence_matrix(const cv::Mat_<unsigned char>& ori, int dx, int dy) const
{
    cv::Mat_<unsigned char> mat = cv::Mat_<unsigned char>::zeros(height, width);
    for(int y = 1; y < height - 1; ++y)
    {
        for(int x = 1; x < width - 1; ++x)
        {
            const int tx = x + dx;
            const int ty = y + dy;
            if(tx < 0 || width <= tx || ty < 0 || height <= ty) continue;

            const unsigned char here  = ori(y, x);
            const unsigned char other = ori(ty, tx);
            if(here < 0 || other < 0) continue;
            mat(y, x) = static_cast<unsigned char>(here * param_cohog.BinCount + other);
        }
    }
    return mat;
}

std::vector<cv::Mat_<unsigned char> > CoHOGDetector2::calc_cooccurrence_matrices(const cv::Mat_<unsigned char>& ori) const
{
    std::vector<cv::Mat_<unsigned char> > matrices(n_offset);
    for(int i = 0; i < n_offset; ++i)
        matrices[i] = calc_cooccurrence_matrix(ori, offsets_x[i], offsets_y[i]);
    return matrices;
}

std::vector<Window> CoHOGDetector2::detect(const cv::Mat_<unsigned char>& img)
{
    assert(img.cols == width && img.rows == height);

    const cv::Mat_<unsigned char> orientation                   = calc_gradient_orientation_matrix(img, param_cohog.BinCount, param_cohog.MinGradient);
    const std::vector<cv::Mat_<unsigned char> > occur_matrices  = calc_cooccurrence_matrices(orientation);

    return std::vector<Window>();
}

}

