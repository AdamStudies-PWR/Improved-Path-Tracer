#pragma once

#include <random>
#include <vector>
#include <utility>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"

#include "scene/objects/EReflectionType.hpp"

namespace tracer::scene::objects
{

typedef std::vector<std::pair<containers::Ray, double>> RayData;

class AObject
{
public:
    AObject(containers::Vec3 position, containers::Vec3 emission, containers::Vec3 color, EReflectionType reflection);

    virtual double intersect(const containers::Ray& ray) const = 0;
    virtual RayData calculateReflections(const containers::Ray& ray, const containers::Vec3& intersection,
        const uint16_t depth, std::mt19937& generator) const = 0;

    containers::Vec3 getColor() const;
    containers::Vec3 getEmission() const;
    containers::Vec3 getPosition() const;
    EReflectionType getReflectionType() const;

protected:
    containers::Vec3 color_;
    containers::Vec3 emission_;
    containers::Vec3 position_;
    EReflectionType reflection_;
};

}  // namespace tracer::scene::objects
