#include "RendererCPU.hpp"

#include <cstdio>
#include <iostream>

#include <omp.h>


namespace tracer::renderer
{

void renderCPU(const uint32_t height, std::function<void(const uint32_t)> gpuCallback)
{
    unsigned counter = 0;
    fprintf(stdout, "Rendering %.2f%%", (float)counter);
    fflush(stdout);

    #pragma omp parallel for
    for (uint32_t z = 0; z < height; z++)
    {
        gpuCallback(z);
        counter++;
        fprintf(stdout, "\rRendering %.2f%%", ((float)counter/(height)*100));
    }
    fflush(stdout);
}

}  // namespace tracer::renderer
