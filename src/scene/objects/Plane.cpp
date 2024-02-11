#include "Plane.hpp"

#include <iostream>

#include "math.h"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;

const double MARGIN = 1e-4;

double distanceToBorder(const Vec3& origin, const Vec3& border, const Vec3& impact)
{
    const auto refPoint = impact - origin;
    const auto top = border.xx_ * refPoint.xx_ + border.yy_ * refPoint.yy_ + border.zz_ * refPoint.zz_;
    const auto bottom = pow(border.xx_, 2) + pow(border.yy_, 2) + pow(border.zz_, 2);

    if (bottom == 0.0) return 0.0;

    const auto distance = top / bottom;
    return (origin + border * distance).distance(impact);
}
}

Plane::Plane(Vec3 north, Vec3 east, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
{
    planeVector_ = (north % east).norm();

    bottomRight_ = position_ + (north * -1) + east;
    bottomLeft_ = position_ + (north * -1) + (east * -1);
    topLeft_ = position_ + north + (east * -1);
    topRight_ = position_ + north + east;

    distanceHorizontal_ = bottomLeft_.distance(bottomRight_);
    distanceVertical_ = bottomLeft_.distance(topLeft_);
}

double Plane::intersect(const Ray& ray) const
{
    const auto refPoint = position_ - ray.origin_;
    const auto top = planeVector_.xx_ * refPoint.xx_ + planeVector_.yy_ * refPoint.yy_
        + planeVector_.zz_ * refPoint.zz_;
    const auto bottom = planeVector_.xx_ * ray.direction_.xx_ + planeVector_.yy_ * ray.direction_.yy_
        + planeVector_.zz_ * ray.direction_.zz_;

    if (bottom == 0.0) return 0.0;

    auto distance = top/bottom;
    if (distance <= 0.0) return 0.0;

    const auto impact = ray.origin_ + (ray.direction_ * distance);
    return checkIfInBounds(impact) ? distance : 0.0;
}

RayData Plane::calculateReflections(const Vec3& intersection, const Vec3& incoming, std::mt19937& generator,
    const uint8_t depth) const
{
    const auto normal = (incoming.dot(planeVector_) < 0 ? planeVector_ * -1 : planeVector_) * -1;

    switch (reflection_)
    {
    case Specular: return handleSpecular(intersection, incoming, normal, generator, depth);
    case Diffuse: return handleDiffuse(intersection, normal, generator);
    case Refractive: return handleRefractive(intersection, incoming, normal, generator, depth);
    default: std::cout << "Uknown reflection type" << std::endl;
    }

    return {};
}

bool Plane::checkIfInBounds(const Vec3& impact) const
{
    auto vertical = distanceToBorder(bottomLeft_, (bottomLeft_ - bottomRight_).norm(), impact);
    if (distanceVertical_ - vertical < -MARGIN) return false;
    vertical = vertical + distanceToBorder(topLeft_, (topLeft_ - topRight_).norm(), impact);
    if (distanceVertical_  - vertical < -MARGIN or distanceVertical_  - vertical > MARGIN) return false;

    auto horizontal = distanceToBorder(bottomLeft_, (bottomLeft_ - topLeft_).norm(), impact);
    if (distanceHorizontal_ - horizontal < -MARGIN) return false;
    horizontal = horizontal + distanceToBorder(bottomRight_, (bottomRight_ - topRight_).norm(), impact);
    if (distanceHorizontal_ - horizontal < -MARGIN or distanceHorizontal_ - horizontal > MARGIN) return false;

    return true;
}

}  // namespace tracer::scene::objects
