#pragma once

#include "objects/EReflectionType.hpp"
#include "utils/Ray.hpp"
#include "utils/Vec.hpp"

namespace tracer::objects
{

class Sphere
{
public:
    Sphere(double radius, utils::Vec position, utils::Vec emission, utils::Vec color, EReflectionType relfection);

    double intersect(const utils::Ray& ray) const;

    double radius_;
    utils::Vec position_;
    utils::Vec emission_;
    utils::Vec color_;
    EReflectionType relfection_;
};

}  // namespace tracer::objects
