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

std::uniform_real_distribution<> zero_one(-1.0, 1.0);
}  // namespace

Sphere::Sphere(double radius, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
    , radius_(radius)
{}

double Sphere::intersect(const Ray& ray) const
{
    double intersection = 0.0;

    Vec3 op = ray.origin_ - position_;
    double b = op.dot(ray.direction_);
    double delta = b*b - op.dot(op) + radius_*radius_;

    if (delta < 0) return 0;
    else delta = sqrt(delta);

    return (intersection = -b - delta) > MARGIN
        ? intersection : ((intersection = -b + delta) > MARGIN ? intersection : 0);
}

Ray Sphere::calculateReflection(const Ray& ray, const Vec3& intersection, std::mt19937& generator) const
{
    Vec3 normal = (intersection - position_).norm();
    Vec3 surfaceNormal = normal.dot(ray.direction_) < 0 ? normal : normal * -1;

    if (reflection_ == Diffuse)
    {
        auto angle = 2 * M_PI * zero_one(generator);

        std::cout << angle << ", " << surfaceNormal << std::endl;
    }

    return Ray();
}

}  // namespace tracer::scene::objects
