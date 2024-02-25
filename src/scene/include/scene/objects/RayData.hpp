#pragma once

#include "containers/Ray.hpp"

namespace tracer::scene::objects
{

#ifndef __device__
#define __device__
#endif

struct RayData
{
    __device__ RayData(containers::Ray ray=containers::Ray(), double power=0.0,
        containers::Ray secondRay=containers::Ray(), double secondPower=0.0, bool useSecond=false)
        : ray_(ray)
        , secondRay_(secondRay)
        , power_(power)
        , secondPower_(secondPower)
        , useSecond_(useSecond)
    {}

    containers::Ray ray_;
    containers::Ray secondRay_;
    double power_;
    double secondPower_;
    bool useSecond_;
};

}  // namespace tracer::scene::objects
