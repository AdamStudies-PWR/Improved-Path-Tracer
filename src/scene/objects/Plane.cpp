#include "Plane.hpp"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}

Plane::Plane(Vec3 north, Vec3 east, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
    , north_(north)
    , east_(east)
{}

double Plane::intersect(const Ray&) const
{
    double intersection = 0.0;

    return intersection;
}

Ray Plane::getReflectedRay(const Ray&, const Vec3&) const
{
    return Ray();
}

}  // namespace tracer::scene::objects
