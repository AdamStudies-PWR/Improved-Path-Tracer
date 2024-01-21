#pragma once

#include "scene/objects/AObject.hpp"

namespace tracer::scene::objects
{

class Plane : public AObject
{
public:
    Plane(containers::Vec3 north, containers::Vec3 east, containers::Vec3 position, containers::Vec3 emission,
        containers::Vec3 color, EReflectionType reflection);

    double intersect(const containers::Ray& ray) const override;
    containers::Ray getReflectedRay(const containers::Ray& ray, const containers::Vec3& intersection) const override;

private:
    bool checkIfInBounds(const containers::Vec3& impact) const;

    containers::Vec3 bottomLeft_;
    containers::Vec3 bottomRight_;
    containers::Vec3 planeVector_;
    containers::Vec3 topLeft_;
    containers::Vec3 topRight_;
    double distanceHorizontal_;
    double distanceVertical_;
};

}  // namespace tracer::scene::objects
