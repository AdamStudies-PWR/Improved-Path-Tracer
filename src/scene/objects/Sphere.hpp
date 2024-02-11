#pragma once

#include "scene/objects/AObject.hpp"

namespace tracer::scene::objects
{

class Sphere : public AObject
{
public:
    Sphere(double radius, containers::Vec3 position, containers::Vec3 emission, containers::Vec3 color,
        EReflectionType reflection);

    double intersect(const containers::Ray& ray) const override;
    RayData calculateReflections(const containers::Vec3& intersection, const containers::Vec3& incoming,
        std::mt19937& generator, const uint8_t depth) const override;

private:
    double radius_;
};

}  // namespace tracer::scene::objects
