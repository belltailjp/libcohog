#include <libcohog/CoHOGDetector.hpp>
#include <stdexcept>

namespace libcohog
{

void CoHOGDetector::set_detector(const std::vector<double>& _weights)
{
    if(param_cohog.dimension() != _weights.size())
        throw std::invalid_argument("The dimension of given weight vector is different from the dimension of CoHOG feature");

    weights = _weights;
}

void CoHOGDetector::set_detector(model *liblinear_model)
{
    const int dim = liblinear_model->nr_feature;
    std::vector<double> weights(dim, 0);
    for(int idx = 0; idx < dim; ++idx)
        weights[idx] = liblinear_model->w[idx];
    set_detector(weights);
}

void CoHOGDetector::set_detector(const char* liblinear_model_file)
{
    model *m = load_model(liblinear_model_file);
    set_detector(m);
    free_and_destroy_model(&m);
}


int CoHOGDetector::quantitize_gradient(int level, float th, int dx, int dy) const
{
    if(static_cast<float>(dx * dx + dy * dy) < th * th)
        return -1;

    const double rad = std::atan2(dy, dx);
    const int deg    = static_cast<int>(rad * 180.0 / M_PI);
    const int quant  = static_cast<int>(std::floor(deg * level / 360.0 + 0.5));
    const int norm   = (quant + 2 + level) % level;
    return norm;
}

cv::Mat_<unsigned char> CoHOGDetector::calc_gradient_orientation_matrix(const cv::Mat_<unsigned char>& image, unsigned level, float th) const
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


std::vector<float> CoHOGDetector::calculate_feature(const cv::Mat_<unsigned char>& image)
{
    if(param_cohog.width() != image.cols || param_cohog.height() != image.rows)
        throw std::invalid_argument("The CoHOG window size calculated by \"params\" missmatches with the \"image\".");

    const cv::Mat_<unsigned char> orientation = calc_gradient_orientation_matrix(image, param_cohog.BinCount, param_cohog.MinGradient);
    const unsigned w = image.cols;
    const unsigned h = image.rows;
    
    const unsigned dim_per_block  = param_cohog.BinCount * param_cohog.BinCount;
    const unsigned dim_per_offset = dim_per_block  * param_cohog.BlockCountX * param_cohog.BlockCountY;
    const unsigned dim_per_cohog  = dim_per_offset * n_offset;

    std::vector<float> data(dim_per_cohog);
    std::fill(data.begin(), data.end(), 0);

    for(unsigned i = 0; i < n_offset; ++i)
    {
        const int ofst_x         = offsets_x[i];
        const int ofst_y         = offsets_y[i];
        const unsigned begin_idx = i * dim_per_offset;

        for(unsigned blockY = 0; blockY < param_cohog.BlockCountY; ++blockY)
        {
            for(unsigned blockX = 0; blockX < param_cohog.BlockCountX; ++blockX)
            {
                const unsigned idx_block       = blockY * param_cohog.BlockCountX + blockX;
                const unsigned begin_idx_block = begin_idx + idx_block * dim_per_block;

                const unsigned beginX = blockX * param_cohog.BlockSize;
                const unsigned beginY = blockY * param_cohog.BlockSize;
                const unsigned endX   = beginX + param_cohog.BlockSize;
                const unsigned endY   = beginY + param_cohog.BlockSize;

                for(unsigned y = beginY; y < endY; ++y)
                {
                    for(unsigned x = beginX; x < endX; ++x)
                    {
                        const int _x = x + ofst_x;
                        const int _y = y + ofst_y;
                        if(_x < 0 || w <= _x || _y < 0 || h <= _y)
                            continue;

                        const unsigned char val_center = orientation( y,  x);
                        const unsigned char val_offset = orientation(_y, _x);

                        if(val_center != 0xff && val_offset != 0xff)
                        data[begin_idx_block + val_center * param_cohog.BinCount + val_offset] += 1;
                    }
                }
            }
        }
    }
    return data;
}


std::vector<Window> CoHOGDetector::detect(const cv::Mat_<unsigned char>& img)
{
    const int w = img.cols;
    const int h = img.rows;
    const int w_window = param_cohog.width();
    const int h_window = param_cohog.height();

    std::vector<Window> result;

#ifdef WITH_OMP
    #pragma omp parallel for
#endif
    for(int y = 0; y < h - h_window; y += param_scan.SkipSizeY)
    {
        for(int x = 0; x < w - w_window; x += param_scan.SkipSizeX)
        {
            const cv::Mat_<unsigned char> img_clipped = img.rowRange(y, y + h_window).colRange(x, x + w_window);
            const std::vector<float> feature = calculate_feature(img_clipped);

            // caluclate the svm score of the feature
            double score = 0;
            for(int i = 0; i < feature.size(); ++i)
                score += feature[i] * weights[i];

            Window w;
            w.x = x;
            w.y = y;
            w.w = w_window;
            w.h = h_window;
            w.v = score;
#ifdef WITH_OMP
            #pragma omp critical
#endif
            {
                result.push_back(w);
            }
        }
    }

    std::sort(result.rbegin(), result.rend());

    return result;
}

}

