#include "RendererCPU.hpp"

#include <cstdio>
#include <iostream>

#include <omp.h>


namespace tracer::renderer
{

void renderCPU(const uint32_t height, std::function<void(const uint32_t)> gpuCallback)
{
    #pragma omp parallel for num_threads(height)
    for (uint32_t z = 0; z < height; z++)
    {
        gpuCallback(z);
    }
}

}  // namespace tracer::renderer
