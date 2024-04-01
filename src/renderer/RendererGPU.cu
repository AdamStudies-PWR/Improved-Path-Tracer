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
const float FOV_SCALE = 0.0009;
const double INF = 1e20;

__device__ inline LoopRange caculateRange(const uint32_t id, const uint32_t width)
{
    auto range = width / THREAD_LIMIT;
    const auto overflow = width % THREAD_LIMIT;
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

__device__ inline Vec3 deepLayers(AObject** objects, const uint32_t objectsCount, Ray ray, uint8_t depth,
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
    for (int8_t i=(depth - 2); i>= 0; i--)
    {
        pixel = objectEmissions[i] + objectColors[i].mult(pixel);
    }

    delete objectEmissions;
    delete objectColors;

    return pixel;
}

__device__ inline Vec3 secondLayer(AObject** objects, const uint32_t objectsCount, Ray ray, uint8_t& depth,
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

__device__ inline Vec3 firstLayer(AObject** objects, const uint32_t objectsCount, Ray ray, const uint32_t maxDepth,
    curandState& state)
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

__device__ inline Vec3 probePixel(AObject** objects, const uint32_t pixelX, const uint32_t pixelZ, const Vec3& vecX,
    const Vec3& vecZ, const Vec3& center, const Vec3& direction, const uint32_t width, const uint32_t height,
    const uint32_t samples, const uint32_t objectsCount, const uint32_t maxDepth, curandState& state)
{
    auto correctionX = (width % 2 == 0) ? 0.5 : 0.0;
    auto correctionZ = (height % 2 == 0) ? 0.5 : 0.0;
    double stepX = (pixelX < width/2)
        ? width/2 - pixelX - correctionX
        : ((double)width/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
    double stepZ = (pixelZ < height/2)
        ? height/2 - pixelZ - correctionZ
        : ((double)height/2 - pixelZ - 1.0) + ((correctionZ == 0.0) ? 1.0 : correctionZ);

    const auto gaze = (direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE).norm();

    auto pixel = Vec3();
    for (uint32_t i=0; i<samples; i++)
    {
        const auto xFactor = tent_filter(state);
        const auto zFactor = tent_filter(state);
        const auto tentFilter = vecX * xFactor + vecZ * zFactor;

        const auto origin = center + vecX*stepX + vecZ*stepZ + tentFilter;
        pixel = pixel + firstLayer(objects, objectsCount, Ray(origin + direction * VIEWPORT_DISTANCE, gaze), maxDepth,
            state);
    }

    pixel.xx_ = pixel.xx_/samples;
    pixel.yy_ = pixel.yy_/samples;
    pixel.zz_ = pixel.zz_/samples;

    return pixel;
}
}  // namespace

__global__ void cudaMain(Vec3* row, AObject** objects, SceneConstants* constants, uint32_t z)
{
    __shared__ AObject* sharedObjects[MAX_OBJECT_COUNT];
    const auto id = threadIdx.x;

    const auto limit = (THREAD_LIMIT < constants->width_) ? THREAD_LIMIT : constants->width_;
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

    curandState state;
    auto seed = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(123456, seed, 0, &state);

    const auto range = caculateRange(id, constants->width_);

    for (uint32_t x=range.start_; x<range.stop_; x++)
    {
        row[x] = probePixel(sharedObjects, x, z, constants->vecX_, constants->vecZ_, constants->center_,
            constants->direction_, constants->width_, constants->height_, constants->samples_, constants->objectCount_,
            constants->maxDepth_, state);
    }
}

}  // namespace tracer::render
