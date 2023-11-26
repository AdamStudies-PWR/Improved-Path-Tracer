#pragma once

#include "containers/Vec.hpp"

namespace tracer::containers
{

class Ray
{
public:
    Ray(Vec oo=Vec(), Vec dd=Vec());

    Vec oo_;
    Vec dd_;
};

}  // namespace tracer::containers
