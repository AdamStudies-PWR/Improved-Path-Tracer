#pragma once

#include <random>
#include <vector>
#include <utility>

#include <curand_kernel.h>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/objects/EReflectionType.hpp"
#include "scene/objects/RayData.hpp"


namespace tracer::scene::objects
{

#ifndef __device__
#define __device__
#endif

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
    __device__ virtual RayData calculateReflections(const containers::Vec3& intersection,
        const containers::Vec3& incoming, curandState& state, const uint8_t depth) const = 0;

    __device__ containers::Vec3 getEmission() const { return emission_; }
    __device__ containers::Vec3 getColor() const { return color_; }
    __device__ containers::Vec3 getPosition() const { return position_; }
    __device__ EReflectionType getReflectionType() const { return reflection_; }

protected:
    containers::Vec3 color_;
    containers::Vec3 emission_;
    containers::Vec3 position_;
    EReflectionType reflection_;
};

}  // namespace tracer::scene::objects
