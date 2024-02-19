#pragma once

#include <curand_kernel.h>

namespace tracer::utils
{

#ifndef __device__
#define __device__
#endif

bool checkCudaSupport();

__device__ double one_one(auto& state)
{
    return (curand_uniform_double(&state) * 2) - 1;
}

__device__ double tent_filter(auto& state)
{
    return one_one(state);
}


}  // namespace tracer::utils
