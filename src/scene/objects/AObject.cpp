#include "scene/objects/AObject.hpp"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}

AObject::AObject(Vec position, Vec emission, Vec color, EReflectionType relfection)
    : color_(color)
    , emission_(emission)
    , position_(position)
    , reflection_(reflection_)
{}

Vec AObject::getColor() const { return color_;}
Vec AObject::getEmission() const { return emission_; }
Vec AObject::getPosition() const { return position_; }
EReflectionType AObject::getReflectionType() const { return reflection_; }

}  // namespace tracer::scene::objects
