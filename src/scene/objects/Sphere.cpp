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
    Vec3 normal = (intersection - position_).norm();
    Vec3 surfaceNormal = normal.dot(ray.direction_) < 0 ? normal : normal * -1;

    switch (reflection_)
    {
    case Diffuse: return {{Ray(intersection, calculateDiffuseDirection(surfaceNormal, generator)), 1.0}};
    case Specular: return {{Ray(intersection, calculateSpecularDirection(ray.direction_, normal)), 1.0}};
    case Refractive: return calculateRefractive(ray, intersection, depth, normal, surfaceNormal, generator);
    default: std::cout << "Uknown reflection type" << std::endl;
    }

    return {};
}

Vec3 Sphere::calculateDiffuseDirection(Vec3& surfaceNormal, std::mt19937& generator) const
{
    auto angle = 2 * M_PI * zero_one(generator);
    auto distance = zero_one(generator);
    auto ortX = ((fabs(surfaceNormal.xx_) > 0.1 ? Vec3(0, 1, 0) : Vec3(1, 0, 0))%surfaceNormal).norm();
    auto ortY = surfaceNormal%ortX;

    return (ortX*cos(angle)*sqrt(distance) + ortY*sin(angle)*sqrt(distance) + surfaceNormal*sqrt(1 - distance)).norm();
}

Vec3 Sphere::calculateSpecularDirection(const Vec3& direction, const Vec3& normal) const
{
    return direction - normal * 2 * normal.dot(direction);
}

RayData Sphere::calculateRefractive(const Ray& ray, const Vec3& intersection, const uint16_t depth,
    const Vec3& normal, const Vec3& surfaceNormal, std::mt19937& generator) const
{
    Ray reflected = Ray(intersection, calculateSpecularDirection(ray.direction_, normal));
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
