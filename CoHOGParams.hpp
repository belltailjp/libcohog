#pragma once

namespace libcohog
{

static const int offsets_x[] = {0, 1, 2, 3, 4, -4, -3, -2, -1, 0, 1, 2, 3, 4, -3, -2, -1, 0, 1, 2, 3, -3, -2, -1, 0, 1, 2, 3, -1, 0, 1};
static const int offsets_y[] = {0, 0, 0, 0, 0,  1,  1,  1,  1, 1, 1, 1, 1, 1,  2,  2,  2, 2, 2, 2, 2,  3,  3,  3, 3, 3, 3, 3,  4, 4, 4};
static const int n_offset    = sizeof(offsets_x) / sizeof(int);

/*!
 * Describes the parameters of CoHOG feature extraction.
 */
struct CoHOGParams
{
    static const unsigned DefaultBinCount      = 8;
    static const unsigned DefaultBlockSize     = 6;
    static const float    DefaultMinGradient   = 10;
    static const unsigned DefaultBlockCountX   = 3;
    static const unsigned DefaultBlockCountY   = 6;

    /*!
     * The number of quantization level of orientation.
     */
    unsigned BinCount;

    /*!
     * The minimum norm of gradient (sqrt(dx**2 + dy**2) where the (x,y) components of gradient are (dx,dy)) that is not to be ignored.
     */
    float MinGradient;

    /*!
     * The horizontal count of blocks on a CoHOG window.
     */
    unsigned BlockCountX;

    /*!
     * The vertical count of blocks on a CoHOG window.
     */
    unsigned BlockCountY;

    /*!
     * The width and height of CoHOG block in pixel.
     */
    unsigned BlockSize;

    /*!
     * Calculate the width of CoHOG window in pixel.
     */
    unsigned width() const { return BlockCountX * BlockSize; }

    /*!
     * Calculate the height of CoHOG window in pixel.
     */
    unsigned height() const { return BlockCountY * BlockSize; }

    /*!
     * Calclate the dimension of CoHOG feature come from this parameters
     */
    unsigned dimension() const { return BinCount * BinCount * BlockCountX * BlockCountY * n_offset; }

    /*!
     * Construct an default parameters
     */
    CoHOGParams():
        BinCount(DefaultBinCount),
        MinGradient(DefaultMinGradient),
        BlockCountX(DefaultBlockCountX),
        BlockCountY(DefaultBlockCountY),
        BlockSize(DefaultBlockSize)
    {
    }
};


/*!
 * Describes the parameters of detection window search
 */
struct ScanParams
{
    static const float    DefaultMinScale      = 2.0;
    static const float    DefaultMaxScale      = 15.0;
    static const float    DefaultScaleFactor   = 1.1;
    static const unsigned DefaultSkipSizeX     = 2;
    static const unsigned DefaultSkipSizeY     = 4;

    /*!
     * The minimum scale of multiple scale search
     */
    float MinScale;

    /*!
     * The maximum scale of multiple scale search
     */
    float MaxScale;

    /*!
     * The factor search of maginification scale beginning from BeginScale to EndScale for multiple scale
     */
    float ScaleFactor;

    /*!
     * The X shift size of sliding window search
     */
    unsigned SkipSizeX;

    /*!
     * The Y shift size of sliding window search
     */
    unsigned SkipSizeY;

    /*!
     * Construct an default parameters
     */
    ScanParams():
        MinScale(DefaultMinScale),
        MaxScale(DefaultMaxScale),
        ScaleFactor(DefaultScaleFactor),
        SkipSizeX(DefaultSkipSizeX),
        SkipSizeY(DefaultSkipSizeY)
    {
    }
};

}

