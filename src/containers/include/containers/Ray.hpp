#pragma once

#include "containers/Vec3.hpp"

namespace tracer::containers
{

class Ray
{
public:
    Ray(Vec3 origin=Vec3(), Vec3 direction=Vec3());

    friend std::ostream& operator<<(std::ostream& os, const Ray& ray);

    Vec3 origin_;
    Vec3 direction_;
};

}  // namespace tracer::containers
