#include "utils/Image.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <math.h>
#include <ranges>
#include <string>

#include <Magick++.h>

namespace tracer::utils
{

namespace
{
using namespace Magick;
using namespace containers;

int toRgb(double x)
{
    return std::clamp(int(x * 255), 0, 255);
}

void getRgbArray(unsigned char* target, const std::vector<Vec3> image)
{
    int i = 0;
    for (const auto pixel : std::ranges::views::reverse(image))
    {
        target[i] = toRgb(pixel.xx_);
        target[i + 1] = toRgb(pixel.yy_);
        target[i + 2] = toRgb(pixel.zz_);
        i = i + 3;

        if ((long unsigned int)i >= (image.size() * 3)) break;
    }
}
}  // namespace

void saveImage(const std::vector<Vec3>& image, const uint32_t height, const uint32_t width)
{
    std::cout << "Saving Image..." << std::endl;

    if (width*height != image.size())
    {
        std::cout << "Error saving image! Size missmatch!" << std::endl;
        return;
    }

    unsigned char pixelArray[image.size() * 3];
    getRgbArray(pixelArray, image);

    InitializeMagick({});
    Image pngImage;
    pngImage.read(width, height, "RGB", Magick::CharPixel, pixelArray);
    pngImage.write("image.png");
}

}  // namespace tracer::utils
