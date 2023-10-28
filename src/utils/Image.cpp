#include "utils/Image.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TODO: This will require refactoring - I want to save as png
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tracer::utils
{

namespace
{
inline int toInt(double x)
{
    return int(pow(std::clamp(x, 0.0, 1.0), 1/2.2) * 255 + 0.5);
}
}  // namespace

void saveImage(containers::Vec* image, int height, int width)
{
    std::cout << __func__ << " - saving render..." << std::endl;

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i=0; i<(width*height); i++)
    {
        fprintf(f,"%d %d %d ", toInt(image[i].xx_), toInt(image[i].yy_), toInt(image[i].zz_));
    }

    std::cout << __func__ << " - Done" << std::endl;
}

}  // namespace tracer::utils
