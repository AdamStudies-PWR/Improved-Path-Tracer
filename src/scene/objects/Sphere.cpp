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

    return (intersection = -b - delta) > MARGIN
        ? intersection : ((intersection = -b + delta) > MARGIN ? intersection : 0.0);
}

RayData Sphere::calculateReflections(const Ray& ray, const Vec3& intersection, const uint16_t depth,
    std::mt19937& generator) const
{
    switch (reflection_)
    {
    case Diffuse: return calculateDiffuse(ray, intersection, generator);
    case Specular: return calculateSpecular(ray, intersection);
    case Refractive: return calculateRefractive(ray, intersection, generator, depth);
    default: std::cout << "Uknown reflection type" << std::endl;
    }

    return {};
}

RayData Sphere::calculateDiffuse(const Ray& ray, const Vec3& intersection, std::mt19937& generator) const
{
    auto normal = (intersection - position_).norm();
    normal = (ray.direction_.dot(normal) < 0 ? normal * -1 : normal) * -1;

    auto direction = Vec3(0, 0, 0);
    while (direction == Vec3(0, 0, 0))
    {
        direction = Vec3(one_one(generator), one_one(generator), one_one(generator));
    }

    direction = direction.norm();
    direction = (direction.dot(normal) < 0) ? direction * -1 : direction;

    return {{Ray(intersection, direction), 1.0}};
}

RayData Sphere::calculateSpecular(const containers::Ray& ray, const containers::Vec3& intersection) const
{
    auto normal = (intersection - position_).norm();
    normal = ray.direction_.dot(normal) < 0 ? normal * -1 : normal;
    auto reflectedDirection = ray.direction_ - normal * 2 * ray.direction_.dot(normal);
    return {{Ray(intersection, reflectedDirection), 1.0}};
}

RayData Sphere::calculateRefractive(const Ray& ray, const Vec3& intersection, std::mt19937& generator,
    const uint16_t depth) const
{
    Vec3 normal = (intersection - position_).norm();
    normal = (ray.direction_.dot(normal) < 0 ? normal * -1 : normal) * -1;

    auto specular = ray.direction_ - normal * 2 * ray.direction_.dot(normal);

    if (depth < 2)
    {
        return {
            {Ray(intersection, specular * -1), 0.92},
            {Ray(intersection, specular), 0.08}
        };
    }

    if (zero_one(generator) > 0.92)
    {
        return {{Ray(intersection, specular), 1.0}};
    }
    else
    {
        return {{Ray(intersection, specular * -1), 1.0}};
    }
}

}  // namespace tracer::scene::objects
