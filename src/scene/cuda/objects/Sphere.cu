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

const uint8_t SPHERE_EXTREMS = 6;
}  // namespace

class Sphere : public AObject
{
public:
    __device__ Sphere(double radius, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
        : AObject(position, emission, color, reflection, SPHERE_EXTREMS)
        , radius_(radius)
    {
        extremes_ = (Vec3*)malloc(sizeof(Vec3) * SPHERE_EXTREMS);
        extremes_[0] = position_ + Vec3(1, 0, 0) * radius_;
        extremes_[1] = position_ - Vec3(1, 0, 0) * radius_;
        extremes_[2] = position_ + Vec3(0, 1, 0) * radius_;
        extremes_[3] = position_ - Vec3(0, 1, 0) * radius_;
        extremes_[4] = position_ + Vec3(0, 0, 1) * radius_;
        extremes_[5] = position_ - Vec3(0, 0, 1) * radius_;
    }

    __device__ ~Sphere()
    {
        free(extremes_);
    }

    __device__ double intersect(const Ray& ray) const override
    {
        double intersection = 0.0;

        Vec3 op = ray.origin_ - position_;
        double b = op.dot(ray.direction_);
        double delta = b*b - op.dot(op) + radius_*radius_;

        if (delta < 0) return 0.0;
        else delta = sqrt(delta);

        return ((intersection = -b - delta) > MARGIN)
            ? intersection : (((intersection = -b + delta) > MARGIN) ? intersection : 0.0);
        return 0.0;
    }

    __device__ RayData calculateReflections(const Vec3& intersection, const Vec3& incoming, curandState& state,
        const uint8_t depth) const override
    {
        const auto rawNormal = (intersection - position_).norm();
        const auto normal = incoming.dot(rawNormal) < 0 ? rawNormal * -1 : rawNormal;

        switch (reflection_)
        {
        case Specular: return handleSpecular(intersection, incoming, normal, state, depth);
        case Diffuse: return handleDiffuse(intersection, normal, state);
        case Refractive: return handleRefractive(intersection, incoming, rawNormal, normal, state, depth);
        default: printf("Uknown reflection type\n");
        }

        return {};
    }

    __device__ double getNormal(const Vec3& intersection, const Vec3& incoming) const override
    {
        const auto rawNormal = (intersection - position_).norm();
        return incoming.dot(rawNormal);
    }

    __device__ virtual void sortExtremes(const Vec3& refPoint) const override
    {
        helpers::quickSort(extremes_, refPoint, SPHERE_EXTREMS);
    }

    __device__ double getAngle(const Vec3& intersection, const Vec3& incoming) const override
    {
        const auto rawNormal = (intersection - position_).norm();
        const auto normal = incoming.dot(rawNormal) < 0 ? rawNormal * -1 : rawNormal;
        const auto cos = normal.dot(incoming) / (normal.length() * incoming.length());
        return(M_PI_2 - acos(cos));
    }

private:
    double radius_;
};

}  // namespace tracer::scene::objects
