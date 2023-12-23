#include "Sphere.hpp"

#include "math.h"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}  // namespace

Sphere::Sphere(double radius, Vec position, Vec emission, Vec color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
    , radius_(radius)
{}

double Sphere::intersect(const Ray& ray) const
{
    Vec op = position_ - ray.oo_;
    double temp;
    double eps = 1e-4;
    double b = op.dot(ray.dd_);
    double det = b*b - op.dot(op) + radius_*radius_;

    if (det < 0) return 0;
    else det = sqrt(det);

    return (temp = b - det) > eps ? temp : ((temp = b + det) > eps ? temp : 0);
}

}  // namespace tracer::scene::objects
