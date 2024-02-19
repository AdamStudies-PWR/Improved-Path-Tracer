#pragma once

#include "containers/Ray.hpp"

namespace tracer::scene::objects
{

#ifndef __device__
#define __device__
#endif

struct RayData
{
    __device__ RayData(containers::Ray ray=containers::Ray(), double power=0.0)
        : ray_(ray)
        , power_(power)
    {}

    containers::Ray ray_;
    double power_;
};

}  // namespace tracer::scene::objects
