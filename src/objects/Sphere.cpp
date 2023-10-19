#include "objects/Sphere.hpp"

#include "math.h"

namespace tracer::objects
{

Sphere::Sphere(double radius, utils::Vec position, utils::Vec emission, utils::Vec color, EReflectionType relfection)
    : radius_(radius)
    , position_(position)
    , emission_(emission)
    , color_(color)
    , relfection_(relfection)
{}

double Sphere::intersect(const utils::Ray& ray) const
{
    utils::Vec op = position_ - ray.oo_;
    double temp;
    double eps = 1e-4;
    double b = op.dot(ray.dd_);
    double det = b*b - op.dot(op) + radius_*radius_;

    if (det < 0) return 0;
    else det = sqrt(det);

    return (temp = b - det) > eps ? temp : ((temp = b + det) > eps ? temp : 0);
}

}  // namespace tracer::objects
