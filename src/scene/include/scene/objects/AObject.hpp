#pragma once

#include "containers/Ray.hpp"
#include "containers/Vec.hpp"

#include "scene/objects/EReflectionType.hpp"

namespace tracer::scene::objects
{

class AObject
{
public:
    AObject(containers::Vec position, containers::Vec emission, containers::Vec color, EReflectionType relfection);

    virtual double intersect(const containers::Ray& ray) const = 0;

    containers::Vec getColor() const;
    containers::Vec getEmission() const;
    containers::Vec getPosition() const;
    EReflectionType getReflectionType() const;

protected:
    containers::Vec color_;
    containers::Vec emission_;
    containers::Vec position_;
    EReflectionType reflection_;
};

}  // namespace tracer::scene::objects
