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
#include "helpers/RenderData.cu"


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

    //return Range(50, 200, 52, 202);
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
            // printf("DBG !!!!!!!!! Applying temp: %lf\n", temp);
            distance = temp;
            index = i;
        }
    }

    return HitData(index, distance);
}

__device__ inline Vec3 findLight(const RenderData& data, const AObject* lastObject, const Ray& ray,
    const Vec3& intersection)
{
    const bool isNormalNegative = (lastObject->getNormal(intersection, ray.direction_) < 0);
    auto loc = lastObject->getPosition();
    // printf("Last hit object X: %f, Y: %f, Z: %f\n", loc.xx_, loc.zz_, loc.yy_);
    // printf("Normal is: %d\n", isNormalNegative);
    double distance = INF;
    int32_t smallest = -1;
    Vec3 target;
    Vec3 direction;

    for (uint32_t i = 0; i < data.lightCount_; i++)
    {
        // printf("Processing lightsource %d\n", i);
        auto extremes = data.lights_[i]->getExtremes();
        for (uint8_t j = 0; j < data.lights_[i]->getExtremesCount(); j++)
        {
            // printf("Extreme X:%f, Y:%f. Z:%f\n", extremes[j].xx_, extremes[j].yy_, extremes[j].zz_);
            auto toLight = intersection.distance(extremes[j]);
            // printf("Distance to light is: %f\n", toLight);
            if (toLight < distance)
            {
                // printf("New distance is smaller then: %f\n", distance);
                target = extremes[j];
                direction = (target - intersection).norm();
                if ((lastObject->getNormal(intersection, direction * -1) < 0) == isNormalNegative)
                {
                    // printf("New distance will be applied\n");
                    smallest = i;
                    distance = toLight;
                }
            }
        }
    }

    if (smallest == -1)
    {
        // printf("Did not find valid light!\n");
        return Vec3(1, 0, 0);
    }

    const auto propHitData = getHitObjectAndDistance(data.props_, Ray(intersection, direction), data.propCount_);
    if (propHitData.distance_ < distance && propHitData.distance_ > 0.0)
    {
        // printf("Ray blocked by object %d, distance: %f\n", propHitData.index_, propHitData.distance_);
        auto obj = data.props_[propHitData.index_]->getPosition();
        // printf("X: %f, Y: %f, Z: %f\n", obj.xx_, obj.yy_, obj.zz_);
        // printf("NO BUENO\n");
        return Vec3(1, 0, 0);
    }

    // printf("Light bonus\n");
    // const auto light = data.lights_[smallest];
    // printf("Light source hit!\n");
    return Vec3(0, 1, 0);
    //return light->getEmission() + light->getColor().mult(Vec3());
}

__device__ inline Vec3 deepLayers(const RenderData& data, Ray ray, uint8_t depth)
{
    Vec3* objectEmissions = new Vec3[data.maxDepth_ - 2];
    Vec3* objectColors = new Vec3[data.maxDepth_ - 2];

    for (; depth < data.maxDepth_; depth++)
    {
        const auto propHitData = getHitObjectAndDistance(data.props_, ray, data.propCount_);
        const auto lightHitData = getHitObjectAndDistance(data.lights_, ray, data.lightCount_);
        if (propHitData.index_ == -1 && lightHitData.index_ == -1) break;

        if (lightHitData.distance_ < propHitData.distance_)
        {
            const auto light = data.lights_[lightHitData.index_];
            objectEmissions[depth - 2] = light->getEmission();
            objectColors[depth - 2] = light->getColor();
            depth++;
            break;
        }

        const auto& object = data.props_[propHitData.index_];
        const auto intersection = ray.origin_ + ray.direction_ * propHitData.distance_;
        const auto reflected = object->calculateReflections(intersection, ray.direction_, data.state_, depth);
        ray = reflected.ray_;

        objectEmissions[depth - 2] = object->getEmission();
        objectColors[depth - 2] = object->getColor();
    }

    Vec3 pixel = Vec3();
    for (int8_t i=(depth - 2); i>= 0; i--)
    {
        pixel = objectEmissions[i] + objectColors[i].mult(pixel);
    }

    delete objectEmissions;
    delete objectColors;

    return pixel;
}

__device__ inline Vec3 secondLayer(const RenderData& data, Ray ray, uint8_t& depth)
{
    const auto propHitData = getHitObjectAndDistance(data.props_, ray, data.propCount_);
    const auto lightHitData = getHitObjectAndDistance(data.lights_, ray, data.lightCount_);
    if (propHitData.index_ == -1 && lightHitData.index_ == -1)
    {
        return Vec3();
    }

    if (lightHitData.distance_ < propHitData.distance_)
    {
        const auto light = data.lights_[lightHitData.index_];
        return light->getEmission() + light->getColor().mult(Vec3());
    }

    const auto& object = data.props_[propHitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * propHitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, data.state_, depth);

    depth++;
    Vec3 backData;
    backData = deepLayers(data, reflected.ray_, depth) * reflected.power_;
    if (reflected.useSecond_)
    {
        backData = backData + deepLayers(data, reflected.secondRay_, depth) * reflected.secondPower_;
    }

    // return findLight(data, object, ray, intersection);

    return object->getEmission() + object->getColor().mult(backData);
}

__device__ inline Vec3 firstLayer(const RenderData& data, Ray ray)
{
    uint8_t depth = 0;
    const auto propHitData = getHitObjectAndDistance(data.props_, ray, data.propCount_);
    const auto lightHitData = getHitObjectAndDistance(data.lights_, ray, data.lightCount_);
    if (propHitData.index_ == -1 && lightHitData.index_ == -1)
    {
        //return Vec3(0, 0, 1);
        return Vec3();
    }

    if (lightHitData.distance_ < propHitData.distance_)
    {
        const auto light = data.lights_[lightHitData.index_];
        return light->getEmission() + light->getColor().mult(Vec3());
    }

    const auto& object = data.props_[propHitData.index_];
    const auto intersection = ray.origin_ + ray.direction_ * propHitData.distance_;
    const auto reflected = object->calculateReflections(intersection, ray.direction_, data.state_, depth);

    depth++;
    Vec3 backData;
    backData = secondLayer(data, reflected.ray_, depth) * reflected.power_;
    if (reflected.useSecond_)
    {
        backData = backData + secondLayer(data, reflected.secondRay_, depth) * reflected.secondPower_;
    }

    // return findLight(data, object, ray, intersection);
    // return secondLayer(data, reflected.ray_, depth);

    return object->getEmission() + object->getColor().mult(backData);
}

__device__ inline Vec3 samplePixel(const RenderData& data, const Camera* camera, ImageData* imageProperties,
    const Vec3* vecZ, const uint32_t pixelX, const uint32_t pixelZ)
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
        const auto xFactor = tent_filter(data.state_);
        const auto zFactor = tent_filter(data.state_);
        const auto tentFilter = vecX*xFactor + (*vecZ)*zFactor;
        // Tent filter

        const auto origin = camera->origin_ + vecX*stepX + (*vecZ)*stepZ + tentFilter;
        pixel = pixel + firstLayer(data, Ray(origin + camera->direction_ * VIEWPORT_DISTANCE, gaze));
    }

    pixel.xx_ = pixel.xx_/imageProperties->samples_;
    pixel.yy_ = pixel.yy_/imageProperties->samples_;
    pixel.zz_ = pixel.zz_/imageProperties->samples_;

    return pixel;
}
}  // namespace

__global__ void cudaMain(Vec3* image, AObject** props, AObject** lights, Camera* camera, Vec3* vecZ,
    ImageData* imageProperties)
{
    if (blockIdx.x == 0 and threadIdx.x == 0)
    {
        printf("\rRendering %.2f%%", (float)counter);
    }

    curandState state;
    auto seed = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(123456, seed, 0, &state);

    RenderData data {props, lights, imageProperties->propCount_, imageProperties->lightCount_,
        imageProperties->maxDepth_, state};

    const auto totalPixels = imageProperties->width_ * imageProperties->height_;
    const auto range = calculateRange(threadIdx.x, blockIdx.x, imageProperties->width_, imageProperties->height_);
    for (uint32_t z = range.startZ_; z < range.endZ_; z++)
    {
        for (uint32_t x = range.startX_; x < range.endX_; x++)
        {
            // printf("Now prcessing pixel (%d, %d)\n", x, z);
            const auto index = z * imageProperties->width_ + x;
            image[index] = samplePixel(data, camera, imageProperties, vecZ, x, z);
            atomicAdd(&counter, 1);
            // printf("\n\n");
        }
        printf("\rRendering %.2f%%", ((float)counter/(totalPixels)*100));
    }
}

}  // namespace tracer::render
