#pragma once

#include <ostream>

#include "containers/Vec3.hpp"

namespace tracer::containers
{

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

struct Ray
{
    __host__ __device__ Ray(Vec3 origin=Vec3(), Vec3 direction=Vec3())
        : origin_(origin)
        , direction_(direction)
    {}

    friend std::ostream& operator<<(std::ostream& os, const Ray& ray);

    Vec3 origin_;
    Vec3 direction_;
};

}  // namespace tracer::containers
