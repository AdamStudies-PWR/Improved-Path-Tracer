#include "Sphere.hpp"

#include <iostream>

#include "math.h"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;

const double GLASS_IOR = 1.5;
const double MARGIN = 1e-4;

std::uniform_real_distribution<> zero_one(0.0, 1.0);
std::uniform_real_distribution<> one_one(-1.0, 1.0);
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

    return ((intersection = -b - delta)) > MARGIN
        ? intersection : (((intersection = -b + delta) > MARGIN) ? intersection : 0.0);
}

RayData Sphere::calculateReflections(const Vec3& intersection, const Vec3& incoming, std::mt19937& generator,
    const uint8_t depth) const
{
    const auto rawNormal = (intersection - position_).norm();
    const auto normal = incoming.dot(rawNormal) < 0 ? rawNormal * -1 : rawNormal;

    switch (reflection_)
    {
    case Specular: return handleSpecular(intersection, incoming, normal, generator, depth);
    case Diffuse: return handleDiffuse(intersection, normal, generator);
    case Refractive: return handleRefractive(intersection, incoming, rawNormal, normal, generator, depth);
    default: std::cout << "Uknown reflection type" << std::endl;
    }

    return {};
}

}  // namespace tracer::scene::objects
