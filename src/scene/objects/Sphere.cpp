#include "Sphere.hpp"

#include "math.h"

// debug
#include <iostream>
//

namespace tracer::scene::objects
{

namespace
{
using namespace containers;

const double MARGIN = 1e-4;
}  // namespace

Sphere::Sphere(double radius, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
    , radius_(radius)
{}

double Sphere::intersect(const Ray& ray) const
{
    double intersection = 0;

    Vec3 op = ray.origin_ - position_;
    double b = op.dot(ray.direction_);
    double delta = b*b - op.dot(op) + radius_*radius_;

    if (delta < 0) return 0;
    else delta = sqrt(delta);

    return (intersection = -b - delta) > MARGIN
        ? intersection : ((intersection = -b + delta) > MARGIN ? intersection : 0);
}

Ray Sphere::getReflectedRay(const Ray& ray, const Vec3& intersection) const
{
    auto normal = (intersection - position_).norm();
    normal = (normal.dot(ray.direction_) < 0) ? normal : (normal * -1);

    if (emission_.xx_ != 0)
    {
        std::cout << "Light source hit!" << std::endl;
    }

    // Here should be russian rullete to determine if recursion should be stopped early
    // We are not doing this for now

    return ray;
}

}  // namespace tracer::scene::objects
