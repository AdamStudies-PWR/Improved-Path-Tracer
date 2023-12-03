#include "scene/objects/AObject.hpp"

#include <iostream>

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}  // namespace

AObject::AObject(Vec position, Vec emission, Vec color, EReflectionType reflection)
    : color_(color)
    , emission_(emission)
    , position_(position)
    , reflection_(reflection)
{}

Vec AObject::getColor() const { return color_;}
Vec AObject::getEmission() const { return emission_; }
Vec AObject::getPosition() const { return position_; }
EReflectionType AObject::getReflectionType() const { return reflection_; }

}  // namespace tracer::scene::objects
