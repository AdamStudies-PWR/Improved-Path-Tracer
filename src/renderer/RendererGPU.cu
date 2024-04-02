#include <stdio.h>

#include "containers/Vec3.hpp"
#include "objects/AObject.hpp"
#include "objects/Plane.cu"
#include "objects/Sphere.cu"
#include "scene/objects/Camera.hpp"
#include "scene/objects/ObjectData.hpp"
#include "scene/objects/RayData.hpp"
#include "utils/CudaUtils.hpp"

#include "Constants.hpp"
#include "helpers/HitData.cu"
#include "helpers/LoopRange.cu"
#include "helpers/PixelData.hpp"
#include "helpers/SceneConstants.hpp"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;
using namespace utils;

const uint32_t MAX_OBJECT_COUNT = 100;
const uint16_t VIEWPORT_DISTANCE = 140;
const double INF = 1e20;

__device__ inline LoopRange calculateCoordinates(const uint32_t id, const uint32_t samples)
{
    auto range = samples / THREAD_LIMIT;
    const auto overflow = samples % THREAD_LIMIT;
    const auto start = id * range + ((id < overflow) ? id : overflow);
    range = range + ((id < overflow) ? 1 : 0);

    return LoopRange(start, start + range);
}

__device__ inline HitData getHitObjectAndDistance(AObject** objects, const uint32_t objectsCount,
    const containers::Ray& ray)
{
    int index = -1;
    double distance = INF;

    for (uint32_t i=0; i<objectsCount; i++)
    {
        auto temp = objects[i]->intersect(ray);
        if (temp && temp < distance)
        {
            distance = temp;
            index = i;
        }
    }

    return HitData(index, distance);
}

__device__ inline Vec3 deepLayers(AObject** objects, const uint32_t objectsCount, Ray ray, uint8_t& depth,
    const uint32_t maxDepth, curandState& state)
{
    Vec3* objectEmissions = new Vec3[maxDepth - 2];
    Vec3* objectColors = new Vec3[maxDepth - 2];

    for (; depth<maxDepth; depth++)
    {
        const auto hitData = getHitObjectAndDistance(objects, objectsCount, ray);
        if (hitData.index_ == -1) break;

        const auto& object = objects[hitData.index_];
        const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
        const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);
        ray = reflected.ray_;

        objectEmissions[depth - 2] = object->getEmission();
        objectColors[depth - 2] = object->getColor();
    }

    Vec3 pixel;
    for (int8_t i=(depth - 2); i>=0; i--)
    {
        pixel = objectEmissions[i] + objectColors[i].mult(pixel);
    }

    delete objectEmissions;
    delete objectColors;

    return pixel;
}

__device__ inline Vec3 secondLayer(AObject** objects, const uint32_t objectsCount, const Ray& ray, uint8_t& depth,
    const uint32_t maxDepth, curandState& state)
{
    const auto hitData = getHitObjectAndDistance(objects, objectsCount, ray);
    if (hitData.index_ == -1) return Vec3();

    const auto& object = objects[hitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

    depth++;
    Vec3 backData;
    if (depth < maxDepth)
    {
        backData = deepLayers(objects, objectsCount, reflected.ray_, depth, maxDepth, state) * reflected.power_;
        if (reflected.useSecond_)
        {
            backData = backData
                + deepLayers(objects, objectsCount, reflected.secondRay_, depth, maxDepth, state)
                * reflected.secondPower_;
        }
    }

    return object->getEmission() + object->getColor().mult(backData);
}

__device__ inline Vec3 firstLayer(AObject** objects, const uint32_t objectsCount, const Ray& ray,
    const uint32_t maxDepth, curandState& state)
{
    uint8_t depth = 0;
    const auto hitData = getHitObjectAndDistance(objects, objectsCount, ray);
    if (hitData.index_ == -1)
    {
        return Vec3();
    }

    const auto& object = objects[hitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

    depth++;
    Vec3 backData;
    if (depth < maxDepth)
    {
        backData = secondLayer(objects, objectsCount, reflected.ray_, depth, maxDepth, state) * reflected.power_;
        if (reflected.useSecond_)
        {
            backData = backData
                + secondLayer(objects, objectsCount, reflected.secondRay_, depth, maxDepth, state)
                * reflected.secondPower_;
        }
    }

    return object->getEmission() + object->getColor().mult(backData);
}
}  // namespace

__global__ void cudaMain(Vec3* samples, AObject** objects, SceneConstants* constants, PixelData* pixel,
    curandState *devState)
{
    __shared__ AObject* sharedObjects[MAX_OBJECT_COUNT];
    const auto id = threadIdx.x;

    const auto limit = (THREAD_LIMIT < constants->samples_) ? THREAD_LIMIT : constants->samples_;
    if (id < constants->objectCount_)
    {
        auto assignedObjects = constants->objectCount_/limit;
        const auto overflow = constants->objectCount_ % limit;
        const auto startingPoint = id * assignedObjects + ((id < overflow) ? id : overflow);
        assignedObjects = assignedObjects + ((id < overflow) ? 1 : 0);
        const auto target = startingPoint + assignedObjects;

        for (auto ii = startingPoint; ii < target; ii++)
        {
            sharedObjects[ii] = objects[ii];
        }
    }
    __syncthreads();

    const auto range = calculateCoordinates(id, constants->samples_);
    for (auto i=range.start_; i<range.stop_; i++)
    {
        const auto xFactor = tent_filter(*devState);
        const auto zFactor = tent_filter(*devState);
        const auto tentFilter = constants->vecX_*xFactor + constants->vecZ_*zFactor;

        const auto origin = constants->center_ + constants->vecX_*pixel->stepX_ + constants->vecZ_*pixel->stepZ_
            + tentFilter;
        samples[i] = firstLayer(sharedObjects, constants->objectCount_,
            Ray(origin + constants->direction_ * VIEWPORT_DISTANCE, pixel->gaze_), constants->maxDepth_, *devState);
    }
}

}  // namespace tracer::render
