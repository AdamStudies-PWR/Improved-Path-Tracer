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
    RayData calculateReflections(const containers::Ray& ray, const containers::Vec3& intersection,
        const uint16_t depth, std::mt19937& generator) const override;

private:
    RayData calculateDiffuse(const containers::Ray& ray, const containers::Vec3& intersection,
        std::mt19937& generator) const;
    RayData calculateSpecular(const containers::Ray& ray, const containers::Vec3& intersection) const;
    RayData calculateRefractive(const containers::Ray& ray, const containers::Vec3& intersection,
        std::mt19937& generator, const uint16_t depth) const;

    double radius_;
};

}  // namespace tracer::scene::objects
