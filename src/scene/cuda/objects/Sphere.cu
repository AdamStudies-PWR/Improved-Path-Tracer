#include "AObject.hpp"

#include "math.h"

#include "Common.hpp"
#include "Constants.hpp"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
using namespace utils;
}  // namespace

class Sphere : public AObject
{
public:
    __device__ Sphere(double radius, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
        : AObject(position, emission, color, reflection)
        , radius_(radius)
    {}

    __device__ double intersect(const Ray& ray) const override
    {
        double intersection = 0.0;

        Vec3 op = ray.origin_ - position_;
        double b = op.dot(ray.direction_);
        double delta = b*b - op.dot(op) + radius_*radius_;

        if (delta < 0) return 0;
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

private:
    __device__ RayData handleSpecular(const Vec3& intersection, const Vec3& incoming, const Vec3& normal,
        curandState& state, const uint8_t depth) const
    {
        auto specular = calculateSpecular(incoming, normal);
        auto diffuse = calculateDiffuse(normal, state);

        if (depth < 2)
        {
            if (curand_uniform_double(&state) > 0.9) return {Ray(intersection, diffuse), 0.08};
            else return {Ray(intersection, specular), 0.92};
        }

        if (curand_uniform_double(&state) > 0.9) return {Ray(intersection, diffuse), 1.0};
        else return {Ray(intersection, specular), 1.0};
    }

    __device__ RayData handleDiffuse(const Vec3& intersection, const Vec3& normal, curandState& state) const
    {
        auto diffuse = calculateDiffuse(normal, state);
        return {Ray(intersection, diffuse), 1.0};
    }

    __device__ RayData handleRefractive(const Vec3& intersection, const Vec3& incoming, const Vec3& rawNormal,
        const Vec3& normal, curandState& state, const uint8_t depth) const
    {
        auto specular = calculateSpecular(incoming, normal);

        auto refractive = calculateRefreactive(incoming, rawNormal);

        if (refractive == Vec3())
        {
            return {Ray(intersection, specular), 1.0};
        }

        if (depth < 2)
        {
            if (curand_uniform_double(&state) > 0.9) return {Ray(intersection, specular), 0.05};
            else return {Ray(intersection, refractive), 0.95};
        }

        if (curand_uniform_double(&state) > 0.95)
        {
            return {Ray(intersection, specular), 1.0};
        }
        else
        {
            return {Ray(intersection, refractive), 1.0};
        }
    }

    double radius_;
};

}  // namespace tracer::scene::objects
