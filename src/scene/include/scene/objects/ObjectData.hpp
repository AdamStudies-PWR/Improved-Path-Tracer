#pragma once

#include "containers/Vec3.hpp"
#include "scene/objects/EReflectionType.hpp"

namespace tracer::scene::objects
{

enum EObjectType
{
    SphereData,
    PlaneData
};

struct ObjectData
{
    ObjectData(const EObjectType type, double radius, containers::Vec3 position,
        containers::Vec3 emission, containers::Vec3 color, objects::EReflectionType reflectionType);

    ObjectData(const EObjectType type, containers::Vec3 north, containers::Vec3 east,
        containers::Vec3 position, containers::Vec3 emission, containers::Vec3 color, objects::EReflectionType reflectionType);

    const EObjectType objectType_;
    const double radius_;
    const containers::Vec3 north_;
    const containers::Vec3 east_;
    const containers::Vec3 position_;
    const containers::Vec3 emission_;
    const containers::Vec3 color_;
    const objects::EReflectionType reflectionType_;
};

}  // namespace tracer::scene::objects
