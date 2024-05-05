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


struct PlaneEquastion
{
    __device__ PlaneEquastion(const double A=0.0, const double B=0.0, const double C=0.0, const double D=0.0)
        : A_(A)
        , B_(B)
        , C_(C)
        , D_(D)
    {}

    double A_;
    double B_;
    double C_;
    double D_;
};

__device__ inline double distanceToBorder(const Vec3& start, Vec3 vec, const Vec3& impact, const double& length)
{
    Vec3 toImpact = start - impact;
    const auto top = (vec % toImpact).length();
    return top / length;
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
        const auto D = -((planeVector_.xx_ * position_.xx_) + (planeVector_.yy_ * position_.yy_)
            + (planeVector_.zz_ * position_.zz_));
        equastion_ = {planeVector_.xx_, planeVector_.yy_, planeVector_.zz_, D};

        const auto bottomRight = position_ + (north * -1) + east;
        const auto bottomLeft = position_ + (north * -1) + (east * -1);
        const auto topLeft = position_ + north + (east * -1);
        const auto topRight = position_ + north + east;

        rightSide_ = {bottomRight, topRight - bottomRight};
        leftSide_ = {bottomLeft, topLeft - bottomLeft};
        topSide_ = {topRight, topLeft - topRight};
        bottomSide_ = {bottomRight, bottomLeft - bottomRight};

        distanceHorizontal_ = bottomLeft.distance(bottomRight);
        distanceVertical_ = bottomLeft.distance(topLeft);

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
        auto vertical = distanceToBorder(topSide_.origin_, topSide_.direction_, impact, distanceHorizontal_);
        if (distanceVertical_ - vertical < -MARGIN) return false;
        vertical = vertical + distanceToBorder(bottomSide_.origin_, bottomSide_.direction_, impact,
            distanceHorizontal_);
        if (distanceVertical_  - vertical < -MARGIN or distanceVertical_  - vertical > MARGIN) return false;

        auto horizontal = distanceToBorder(leftSide_.origin_, leftSide_.direction_, impact, distanceVertical_);
        if (distanceHorizontal_ - horizontal < -MARGIN) return false;
        horizontal = horizontal + distanceToBorder(rightSide_.origin_, rightSide_.direction_, impact,
            distanceVertical_);
        if (distanceHorizontal_ - horizontal < -MARGIN or distanceHorizontal_ - horizontal > MARGIN) return false;

        return true;
    }

    __device__ double getAngle(const Vec3& intersection, const Vec3& incoming) const override
    {
        const auto normal = (incoming.dot(planeVector_) < 0 ? planeVector_ * -1 : planeVector_) * -1;
        const auto cos = normal.dot(incoming) / (normal.length() * incoming.length());
        return(M_PI_2 - acos(cos));
    }


    PlaneEquastion equastion_;
    Vec3 planeVector_;
    double distanceHorizontal_;
    double distanceVertical_;
    Ray rightSide_;
    Ray leftSide_;
    Ray topSide_;
    Ray bottomSide_;
};

}  // namespace tracer::scene::objects
