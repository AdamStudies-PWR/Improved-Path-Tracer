#pragma once

#include "scene/objects/AObject.hpp"

namespace tracer::scene::objects
{

class Sphere : public AObject
{
public:
    Sphere(double radius, containers::Vec position, containers::Vec emission, containers::Vec color,
        EReflectionType reflection);

    double intersect(const containers::Ray& ray) const override;

private:
    double radius_;
};

}  // namespace tracer::scene::objects
