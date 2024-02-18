#pragma once

#include <random>
#include <vector>
#include <utility>

#include <curand_kernel.h>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/objects/EReflectionType.hpp"

namespace tracer::scene::objects
{

#ifndef __device__
#define __device__
#endif

struct RayData
{
    __host__ __device__ RayData(containers::Ray ray, double power)
        : ray_(ray)
        , power_(power)
    {}

    containers::Ray ray_;
    double power_;
};

struct RayDataWrapper
{
    __device__ RayDataWrapper(RayData* data, const uint8_t size)
        : data_(data)
        , size_(size)
    {}

    RayData* data_;
    uint8_t size_;
};

class AObject
{
public:
    __device__ AObject(containers::Vec3 position, containers::Vec3 emission, containers::Vec3 color,
        EReflectionType reflection)
        : color_(color)
        , emission_(emission)
        , position_(position)
        , reflection_(reflection)
    {}

    __device__ virtual double intersect(const containers::Ray& ray) const = 0;
    __device__ virtual RayDataWrapper calculateReflections(const containers::Vec3& intersection,
        const containers::Vec3& incoming, curandState& state, const uint8_t depth) const = 0;

    __device__ containers::Vec3 getEmission() const { return emission_; }
    __device__ containers::Vec3 getColor() const { return color_; }
    containers::Vec3 getPosition() const;
    EReflectionType getReflectionType() const;

protected:
    RayData* handleSpecular(const containers::Vec3& intersection, const containers::Vec3& incoming,
        const containers::Vec3& normal, std::mt19937& generator, const uint8_t depth) const;
    RayData handleDiffuse(const containers::Vec3& intersection, const containers::Vec3& normal,
        std::mt19937& generator) const;
    RayData handleRefractive(const containers::Vec3& intersection, const containers::Vec3& incoming,
        const containers::Vec3& rawNormal, const containers::Vec3& normal, std::mt19937& generator,
        const uint8_t depth) const;

    containers::Vec3 color_;
    containers::Vec3 emission_;
    containers::Vec3 position_;
    EReflectionType reflection_;
};

}  // namespace tracer::scene::objects
