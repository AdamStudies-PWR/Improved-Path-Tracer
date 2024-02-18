#include "scene/objects/AObject.hpp"

#include "math.h"

namespace tracer::scene::objects
{

namespace
{
    const double MARGIN_S = 1e-4;
}  // namespace

class Sphere : public AObject
{
public:
    __device__ Sphere(double radius, containers::Vec3 position, containers::Vec3 emission, containers::Vec3 color,
        EReflectionType reflection)
        : AObject(position, emission, color, reflection)
        , radius_(radius)
    {}

    __device__ double intersect(const containers::Ray& ray) const override
    {
        double intersection = 0.0;

        containers::Vec3 op = ray.origin_ - position_;
        double b = op.dot(ray.direction_);
        double delta = b*b - op.dot(op) + radius_*radius_;

        if (delta < 0) return 0;
        else delta = sqrt(delta);

        return (intersection = -b - delta) > MARGIN_S
            ? intersection : ((intersection = -b + delta) > MARGIN_S ? intersection : 0.0);
        return 0.0;
    }

    __device__ RayDataWrapper calculateReflections(const containers::Vec3& intersection,
        const containers::Vec3& incoming, curandState& state, const uint8_t depth) const override
    {
        const auto rawNormal = (intersection - position_).norm();
        const auto normal = incoming.dot(rawNormal) < 0 ? rawNormal * -1 : rawNormal;

        switch (reflection_)
        {
        // case Specular: return handleSpecular(intersection, incoming, normal, generator, depth);
        // case Diffuse: return handleDiffuse(intersection, normal, generator);
        // case Refractive: return handleRefractive(intersection, incoming, rawNormal, normal, generator, depth);
        default: printf("Uknown reflection type\n");
        }

        return RayDataWrapper(nullptr, 0);
    }

private:
    double radius_;
};

}  // namespace tracer::scene::objects
