#include "scene/objects/AObject.hpp"

#include <iostream>

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}  // namespace

AObject::AObject(Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : color_(color)
    , emission_(emission)
    , position_(position)
    , reflection_(reflection)
{}

Vec3 AObject::getColor() const { return color_;}
Vec3 AObject::getEmission() const { return emission_; }
Vec3 AObject::getPosition() const { return position_; }
EReflectionType AObject::getReflectionType() const { return reflection_; }

}  // namespace tracer::scene::objects
