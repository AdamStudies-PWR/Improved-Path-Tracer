#pragma once

#include "containers/Vec.hpp"

namespace tracer::containers
{

class Ray
{
public:
    Ray(Vec oo=Vec(), Vec dd=Vec());

    friend std::ostream& operator<<(std::ostream& os, const Ray& ray);

    Vec oo_;
    Vec dd_;
};

}  // namespace tracer::containers
