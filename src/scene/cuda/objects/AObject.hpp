#pragma once

#include <random>
#include <vector>
#include <utility>

#include <curand_kernel.h>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/objects/EReflectionType.hpp"
#include "scene/objects/RayData.hpp"
#include "utils/CudaUtils.hpp"


namespace tracer::scene::objects
{

#ifndef __device__
#define __device__
#endif

namespace
{
using namespace containers;

const double GLASS_IOR = 1.5;
const double AIR_IOR = 1.0;

__device__ Vec3 calculateSpecular(const Vec3& incoming, const Vec3& normal)
{
    return incoming - normal * incoming.dot(normal) * 2;
}

__device__ Vec3 calculateDiffuse(const Vec3& normal, curandState& state)
{
    auto direction = Vec3(0, 0, 0);
    while (direction == Vec3(0, 0, 0))
    {
        direction = Vec3(utils::one_one(state), utils::one_one(state), utils::one_one(state));
    }

    direction = direction.norm();
    return (direction.dot(normal) < 0) ? direction * -1 : direction;
}

__device__ Vec3 calculateRefreactive(const Vec3& incoming, const Vec3& normal)
{
    const auto index = AIR_IOR/GLASS_IOR;
    const auto cosIncoming = fabs(normal.dot(incoming));
    auto sinRefracted2 = pow(index, 2) * (1.0 - pow(cosIncoming, 2));

    if (sinRefracted2 > 1.0)
    {
        return Vec3();
    }

    const double cosRefracted = sqrt(1.0 - sinRefracted2);
    return incoming * index + normal * (index * cosIncoming - cosRefracted);
}
}  // namespace

class AObject
{
public:
    __device__ AObject(Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection,
            const uint8_t extremesCount)
        : color_(color)
        , emission_(emission)
        , position_(position)
        , reflection_(reflection)
        , extremesCount_(extremesCount)
    {}

    __device__ virtual double intersect(const Ray& ray) const = 0;
    __device__ virtual RayData calculateReflections(const Vec3& intersection, const Vec3& incoming, curandState& state,
        const uint8_t depth) const = 0;
    __device__ virtual double getNormal(const Vec3& intersection, const Vec3& incoming) const = 0;
    __device__ virtual void sortExtremes(const Vec3& refPoint) const = 0;

    __device__ Vec3 getEmission() const { return emission_; }
    __device__ Vec3 getColor() const { return color_; }
    __device__ Vec3 getPosition() const { return position_; }
    __device__ EReflectionType getReflectionType() const { return reflection_; }
    __device__ uint32_t getExtremesCount() const { return extremesCount_; }
    __device__ Vec3* getExtremes() const { return extremes_; }

protected:
    __device__ RayData handleSpecular(const Vec3& intersection, const Vec3& incoming, const Vec3& normal,
        curandState& state, const uint8_t depth) const
    {
        auto specular = calculateSpecular(incoming, normal);
        auto diffuse = calculateDiffuse(normal, state);

        if (depth < 2)
        {
            return {Ray(intersection, specular), 0.92, Ray(intersection, diffuse), 0.08, true};
        }

        if (curand_uniform_double(&state) > 0.9)
        {
            return {Ray(intersection, diffuse), 1.0};
        }
        else
        {
            return {Ray(intersection, specular), 1.0};
        }
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
            return {Ray(intersection, refractive), 0.95, Ray(intersection, specular), 0.05, true};
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

    Vec3 color_;
    Vec3 emission_;
    Vec3 position_;
    EReflectionType reflection_;
    uint8_t extremesCount_;
    Vec3* extremes_;
};

}  // namespace tracer::scene::objects
