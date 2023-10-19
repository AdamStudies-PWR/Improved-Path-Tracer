#pragma once

#include "utils/Vec.hpp"

namespace tracer::utils
{

class Ray
{
public:
    Ray(Vec oo, Vec dd);

    Vec oo_;
    Vec dd_;
};

}  // namespace tracer::utils
