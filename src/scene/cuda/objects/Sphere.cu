#include "AObject.hpp"

#include "math.h"

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

        if (delta < 0.0) return 0.0;
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
    double radius_;
};

}  // namespace tracer::scene::objects
