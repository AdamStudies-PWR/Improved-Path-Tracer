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
        ? intersection : ((intersection = -b + delta) > MARGIN ? intersection : 0);
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
    return {{Ray(intersection, reflectedDirection), 0.9}};
}

RayData Sphere::calculateRefractive(const Ray& ray, const Vec3& intersection, std::mt19937& generator,
    const uint16_t depth) const
{
    Vec3 normal = (intersection - position_).norm();
    Vec3 surfaceNormal = normal.dot(ray.direction_) < 0 ? normal : normal * -1;

    Ray reflected = calculateSpecular(ray, intersection)[0].first; // hack

    bool isEntering = normal.dot(surfaceNormal) > 0;
    const auto localIOR = isEntering ? 1/GLASS_IOR : GLASS_IOR;
    const auto scalar = ray.direction_.dot(surfaceNormal);
    const auto cos2t = 1 - localIOR*localIOR*(1 - scalar * scalar);

    if (cos2t < 0) return {{reflected, 0.0}};

    Ray refracted = (ray.direction_ * localIOR - normal * ((isEntering ? 1 : -1) * (scalar * localIOR + sqrt(cos2t))))
        .norm();

    auto normalReflectence = (GLASS_IOR - 1) * (GLASS_IOR - 1)/((GLASS_IOR + 1) * (GLASS_IOR + 1));
    auto cTheta = 1 - (isEntering ? -localIOR : refracted.direction_.dot(normal));

    auto fresnelReflectence = normalReflectence + (1 - normalReflectence) * pow(cTheta, 5);

    auto probablilty = 0.25 + 0.5 * fresnelReflectence;

    auto reverseFresnel = 1 - fresnelReflectence;
    auto reflectenceProbability = fresnelReflectence/probablilty;
    auto reverseProbability = reverseFresnel/(1 - probablilty);

    if (depth > 2)
    {
        if (zero_one(generator) < probablilty)
        {
            return {{reflected, reflectenceProbability}};
        }
        else
        {
            return {{refracted, reverseProbability}};
        }
    }

    return {{reflected, fresnelReflectence}, {refracted, reverseFresnel}};
}

}  // namespace tracer::scene::objects
