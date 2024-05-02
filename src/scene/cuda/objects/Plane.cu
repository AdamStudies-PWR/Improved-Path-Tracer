#include "AObject.hpp"

#include "math.h"

#include "Constants.hpp"
#include "Helpers.hpp"


namespace tracer::scene::objects
{

namespace
{
using namespace containers;
using namespace utils;

const uint8_t PLANE_EXTREMS = 5;

__device__ double distanceToBorder(const Vec3& origin, const Vec3& border, const Vec3& impact)
{
    const auto refPoint = impact - origin;
    const auto top = border.xx_ * refPoint.xx_ + border.yy_ * refPoint.yy_ + border.zz_ * refPoint.zz_;
    const auto bottom = pow(border.xx_, 2) + pow(border.yy_, 2) + pow(border.zz_, 2);

    if (bottom == 0.0) return 0.0;

    const auto distance = top / bottom;
    return (origin + border * distance).distance(impact);
}
}  // namespace

class Plane : public AObject
{
public:
    __device__ Plane(Vec3 north, Vec3 east, Vec3 position, Vec3 emission, Vec3 color,
            EReflectionType reflection)
        : AObject(position, emission, color, reflection, PLANE_EXTREMS)
    {
        planeVector_ = (north % east).norm();

        bottomRight_ = position_ + (north * -1) + east;
        bottomLeft_ = position_ + (north * -1) + (east * -1);
        topLeft_ = position_ + north + (east * -1);
        topRight_ = position_ + north + east;

        distanceHorizontal_ = bottomLeft_.distance(bottomRight_);
        distanceVertical_ = bottomLeft_.distance(topLeft_);

        extremes_ = (Vec3*)malloc(sizeof(Vec3) * PLANE_EXTREMS);
        extremes_[0] = position_;
        extremes_[1] = position_ + north;
        extremes_[2] = position_ - north;
        extremes_[3] = position_ + east;
        extremes_[4] = position_ - east;
    }

    __device__ ~Plane()
    {
        free(extremes_);
    }

    __device__ double intersect(const Ray& ray) const override
    {
        const auto refPoint = position_ - ray.origin_;
        const auto top = planeVector_.xx_ * refPoint.xx_ + planeVector_.yy_ * refPoint.yy_
            + planeVector_.zz_ * refPoint.zz_;
        const auto bottom = planeVector_.xx_ * ray.direction_.xx_ + planeVector_.yy_ * ray.direction_.yy_
            + planeVector_.zz_ * ray.direction_.zz_;

        if (bottom == 0.0) return 0.0;

        auto distance = top/bottom;
        if (distance <= MARGIN) return 0.0;

        const auto impact = ray.origin_ + (ray.direction_ * distance);
        if (not checkIfInBounds(impact))
        {
            return 0.0;
        }

        return distance;
    }

    __device__ RayData calculateReflections(const Vec3& intersection, const Vec3& incoming,
        curandState& state, const uint8_t depth) const override
    {
        const auto normal = (incoming.dot(planeVector_) < 0 ? planeVector_ * -1 : planeVector_) * -1;

        switch (reflection_)
        {
        case Specular: return handleSpecular(intersection, incoming, normal, state, depth);
        case Diffuse: return handleDiffuse(intersection, normal, state);
        case Refractive: return handleRefractive(intersection, incoming, normal, normal, state, depth);
        default: printf("Uknown reflection type");
        }

        return {};
    }

    __device__ double getNormal(const Vec3&, const Vec3& incoming) const override
    {
        return incoming.dot(planeVector_);
    }

    __device__ virtual void sortExtremes(const Vec3& refPoint) const override
    {
        helpers::quickSort(extremes_, refPoint, PLANE_EXTREMS);
    }

private:
    __device__ bool checkIfInBounds(const Vec3& impact) const
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

    __device__ double getAngle(const Vec3& intersection, const Vec3& incoming) const override
    {
        const auto normal = (incoming.dot(planeVector_) < 0 ? planeVector_ * -1 : planeVector_) * -1;
        const auto cos = normal.dot(incoming) / (normal.length() * incoming.length());
        return(M_PI_2 - acos(cos));
    }


    Vec3 bottomLeft_;
    Vec3 bottomRight_;
    Vec3 planeVector_;
    Vec3 topLeft_;
    Vec3 topRight_;
    double distanceHorizontal_;
    double distanceVertical_;
};

}  // namespace tracer::scene::objects
