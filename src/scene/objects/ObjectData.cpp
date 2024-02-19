#include "scene/objects/ObjectData.hpp"


namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}  // namespace

ObjectData::ObjectData(const EObjectType objectType, double radius, Vec3 position, Vec3 emission, Vec3 color,
    EReflectionType reflectionType)
    : objectType_(objectType)
    , radius_(radius)
    , north_(Vec3())
    , east_(Vec3())
    , position_(position)
    , emission_(emission)
    , color_(color)
    , reflectionType_(reflectionType)
{}

ObjectData::ObjectData(const EObjectType objectType, Vec3 north, Vec3 east, Vec3 position, Vec3 emission, Vec3 color,
    EReflectionType reflectionType)
    : objectType_(objectType)
    , radius_(0.0)
    , north_(north)
    , east_(east)
    , position_(position)
    , emission_(emission)
    , color_(color)
    , reflectionType_(reflectionType)
{}

}  // namespace tracer::scene::objects
