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
#include "helpers/ImageData.hpp"
#include "helpers/Range.cu"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;
using namespace utils;

const float FOV_SCALE = 0.0009;
const uint16_t VIEWPORT_DISTANCE = 140;
const double INF = 1e20;

__device__ uint32_t counter = 0;

__device__ inline Range calculateRange(const uint32_t idX, const uint32_t idZ, const uint32_t width,
    const uint32_t height)
{
    auto widthAssigned = width/THREAD_SIZE;
    const auto widthOverflow = width % THREAD_SIZE;
    const auto startWidth = idX * widthAssigned + ((idX < widthOverflow) ? idX : widthOverflow);
    widthAssigned = widthAssigned + ((idX < widthOverflow) ? 1 : 0);

    auto heightAssigned = height/BLOCK_SIZE;
    const auto heightOverflow = height % BLOCK_SIZE;
    const auto startHeight = idZ * heightAssigned + ((idZ < heightOverflow) ? idZ : heightOverflow);
    heightAssigned = heightAssigned + ((idZ < heightOverflow) ? 1 : 0);

    return Range(startWidth, startHeight, (startWidth + widthAssigned), (startHeight + heightAssigned));
}

__device__ inline HitData getHitObjectAndDistance(AObject** objects, const containers::Ray& ray,
    const uint32_t objectCount)
{
    int index = -1;
    double distance = INF;

    for (uint32_t i = 0; i < objectCount; i++)
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

 __device__ Vec3 deepLayers(AObject** objects, Ray ray, uint8_t depth, const uint32_t objectCount,
    const uint32_t maxDepth, curandState& state)
{
    Vec3* objectEmissions = new Vec3[maxDepth - 2];
    Vec3* objectColors = new Vec3[maxDepth - 2];

    for (; depth < maxDepth; depth++)
    {
        const auto hitData = getHitObjectAndDistance(objects, ray, objectCount);
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

__device__ inline Vec3 secondLayer(AObject** objects, Ray ray, uint8_t& depth, const uint32_t objectCount,
    const uint32_t maxDepth, curandState& state)
{
    const auto hitData = getHitObjectAndDistance(objects, ray, objectCount);
    if (hitData.index_ == -1) return Vec3();

    const auto& object = objects[hitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

    depth++;
    Vec3 backData;
    backData = deepLayers(objects, reflected.ray_, depth, objectCount, maxDepth, state) * reflected.power_;
    if (reflected.useSecond_)
    {
        backData = backData
            + deepLayers(objects, reflected.secondRay_, depth, objectCount, maxDepth, state) * reflected.secondPower_;
    }

    return object->getEmission() + object->getColor().mult(backData);
}

__device__ inline Vec3 firstLayer(AObject** objects, Ray ray, const uint32_t objectCount, const uint32_t maxDepth,
    curandState& state)
{
    uint8_t depth = 0;
    const auto hitData = getHitObjectAndDistance(objects, ray, objectCount);
    if (hitData.index_ == -1)
    {
        return Vec3();
    }

    const auto& object = objects[hitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

    depth++;
    Vec3 backData;
    backData = secondLayer(objects, reflected.ray_, depth, objectCount, maxDepth, state) * reflected.power_;
    if (reflected.useSecond_)
    {
        backData = backData
            + secondLayer(objects, reflected.secondRay_, depth, objectCount, maxDepth, state) * reflected.secondPower_;
    }

    return object->getEmission() + object->getColor().mult(backData);
}

__device__ inline Vec3 samplePixel(AObject** objects, const Camera* camera, ImageData* imageProperties,
    curandState& state, const Vec3* vecZ, const uint32_t pixelX, const uint32_t pixelZ)
    {
        const auto vecX = camera->orientation_;
        const auto width = imageProperties->width_;
        const auto height = imageProperties->height_;

        auto correctionX = (width % 2 == 0) ? 0.5 : 0.0;
        auto correctionZ = (width % 2 == 0) ? 0.5 : 0.0;
        double stepX = (pixelX < width/2)
            ? width/2 - pixelX - correctionX
            : ((double)width/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
        double stepZ = (pixelZ < height/2)
            ? height/2 - pixelZ - correctionZ
            : ((double)height/2 - pixelZ - 1.0) + ((correctionZ == 0.0) ? 1.0 : correctionZ);

        const auto gaze = (camera->direction_ + vecX*stepX*FOV_SCALE + (*vecZ)*stepZ*FOV_SCALE).norm();

        Vec3 pixel = Vec3();
        for (uint32_t i = 0;  i < imageProperties->samples_; i++)
        {
            // Tent filter
            const auto xFactor = tent_filter(state);
            const auto zFactor = tent_filter(state);
            const auto tentFilter = vecX*xFactor + (*vecZ)*zFactor;
            // Tent filter

            const auto origin = camera->origin_ + vecX*stepX + (*vecZ)*stepZ + tentFilter;
            pixel = pixel + firstLayer(objects, Ray(origin + camera->direction_ * VIEWPORT_DISTANCE, gaze),
                imageProperties->objectCount_, imageProperties->maxDepth_, state);
        }

        pixel.xx_ = pixel.xx_/imageProperties->samples_;
        pixel.yy_ = pixel.yy_/imageProperties->samples_;
        pixel.zz_ = pixel.zz_/imageProperties->samples_;

        return pixel;
    }
}  // namespace

__global__ void cudaMain(Vec3* image, AObject** objects, Camera* camera, Vec3* vecZ, ImageData* imageProperties)
{
    if (blockIdx.x == 0 and threadIdx.x == 0)
    {
        printf("\rRendering %.2f%%", (float)counter);
    }

    curandState state;
    auto seed = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(123456, seed, 0, &state);

    const auto totalPixels = imageProperties->width_ * imageProperties->width_;
    const auto range = calculateRange(threadIdx.x, blockIdx.x, imageProperties->width_, imageProperties->height_);
    for (uint32_t z = range.startZ_; z < range.endZ_; z++)
    {
        for (uint32_t x = range.startX_; x < range.endX_; x++)
        {
            const auto index = z * imageProperties->width_ + x;
            image[index] = samplePixel(objects, camera, imageProperties, state, vecZ, x, z);
            atomicAdd(&counter, 1);
        }
        printf("\rRendering %.2f%%", ((float)counter/(totalPixels)*100));
    }
}

}  // namespace tracer::render
