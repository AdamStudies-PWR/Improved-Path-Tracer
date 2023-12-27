#pragma once

#include "containers/Vec3.hpp"

namespace tracer::containers
{

class Ray
{
public:
    Ray(Vec3 oo=Vec3(), Vec3 dd=Vec3());

    friend std::ostream& operator<<(std::ostream& os, const Ray& ray);

    Vec3 oo_;
    Vec3 dd_;
};

}  // namespace tracer::containers
