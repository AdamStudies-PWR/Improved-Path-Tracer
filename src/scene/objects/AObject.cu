#include "scene/objects/AObject.hpp"

#include <optional>
#include <iostream>

#include "math.h"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;

const double GLASS_IOR = 1.5;
const double AIR_IOR = 1.0;

std::uniform_real_distribution<> zero_one(0.0, 1.0);
std::uniform_real_distribution<> one_one(-1.0, 1.0);

Vec3 calculateSpecular(const Vec3& incoming, const Vec3& normal)
{
    return Vec3();//incoming - normal * incoming.dot(normal) * 2;
}

Vec3 calculateDiffuse(const Vec3& normal, std::mt19937& generator)
{
    /*auto direction = Vec3(0, 0, 0);
    while (direction == Vec3(0, 0, 0))
    {
        direction = Vec3(one_one(generator), one_one(generator), one_one(generator));
    }

    direction = direction.norm();
    return (direction.dot(normal) < 0) ? direction * -1 : direction;*/
    return Vec3();
}

std::optional<Vec3> calculateRefreactive(const Vec3& incoming, const Vec3& normal)
{
    /*const auto index = AIR_IOR/GLASS_IOR;
    const auto cosIncoming = fabs(normal.dot(incoming));
    auto sinRefracted2 = pow(index, 2) * (1.0 - pow(cosIncoming, 2));

    if (sinRefracted2 > 1.0)
    {
        return std::nullopt;
    }

    const double cosRefracted = sqrt(1.0 - sinRefracted2);
    return incoming * index + normal * (index * cosIncoming - cosRefracted);*/
    return std::nullopt;
}
}  // namespace

Vec3 AObject::getPosition() const { return position_; }
EReflectionType AObject::getReflectionType() const { return reflection_; }

RayData* AObject::handleSpecular(const Vec3& intersection, const Vec3& incoming, const Vec3& normal,
    std::mt19937& generator, const uint8_t depth) const
{
    /*auto specular = calculateSpecular(incoming, normal);
    auto diffuse = calculateDiffuse(normal, generator);

    if (depth < 2)
    {
        return {
            {Ray(intersection, specular), 0.92},
            {Ray(intersection, diffuse), 0.08}
        };
    }

    if (zero_one(generator) > 0.9)
    {
        return {{Ray(intersection, diffuse), 1.0}};
    }
    else
    {
        return {{Ray(intersection, specular), 1.0}};
    }*/
    return nullptr;
}

RayData AObject::handleDiffuse(const Vec3& intersection, const Vec3& normal, std::mt19937& generator) const
{
    // auto diffuse = calculateDiffuse(normal, generator);
    // return {{Ray(intersection, diffuse), 1.0}};
    return RayData(Ray(), 0.0);
}

RayData AObject::handleRefractive(const Vec3& intersection, const Vec3& incoming, const Vec3& rawNormal,
    const Vec3& normal, std::mt19937& generator, const uint8_t depth) const
{
    /*auto specular = calculateSpecular(incoming, normal);
    auto maybeRefreactive = calculateRefreactive(incoming, rawNormal);

    if (not maybeRefreactive.has_value())
    {
        return {{Ray(intersection, specular), 1.0}};
    }

    if (depth < 2)
    {
        return {
            {Ray(intersection, maybeRefreactive.value()), 0.95},
            {Ray(intersection, specular), 0.05}
        };
    }

    if (zero_one(generator) > 0.95)
    {
        return {{Ray(intersection, specular), 1.0}};
    }
    else
    {
        return {{Ray(intersection, maybeRefreactive.value()), 1.0}};
    }*/
    return RayData(Ray(), 0.0);
}

}  // namespace tracer::scene::objects
