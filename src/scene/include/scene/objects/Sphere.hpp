#pragma once

#include "scene/objects/EReflectionType.hpp"
#include "containers/Ray.hpp"
#include "containers/Vec.hpp"

namespace tracer::objects
{

class Sphere
{
public:
    Sphere(double radius, containers::Vec position, containers::Vec emission, containers::Vec color, EReflectionType relfection);

    double intersect(const containers::Ray& ray) const;

    double radius_;
    containers::Vec position_;
    containers::Vec emission_;
    containers::Vec color_;
    EReflectionType relfection_;
};

}  // namespace tracer::objects
