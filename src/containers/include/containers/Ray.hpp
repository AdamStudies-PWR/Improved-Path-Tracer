#pragma once

#include "containers/Vec.hpp"

namespace tracer::containers
{

class Ray
{
public:
    Ray(Vec oo, Vec dd);

    Vec oo_;
    Vec dd_;
};

}  // namespace tracer::containers
