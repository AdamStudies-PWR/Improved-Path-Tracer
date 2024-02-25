#include <stdio.h>

#include "containers/Vec3.hpp"
#include "objects/AObject.hpp"
#include "objects/Plane.cu"
#include "objects/Sphere.cu"
#include "scene/objects/Camera.hpp"
#include "scene/objects/ObjectData.hpp"
#include "scene/objects/RayData.hpp"
#include "utils/CudaUtils.hpp"

#include "Coordinates.cu"
#include "Constants.hpp"
#include "HitData.cu"


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

__device__ Coordinates calculateCoordinates(const uint32_t idX, const uint32_t idZ,
    const uint32_t width, const uint32_t height)
{
    if (width <= BLOCK_SIZE && height <= BLOCK_SIZE)
    {
        return Coordinates(idX, idZ, 0, 0);
    }

    const uint32_t xAddition = width % BLOCK_SIZE;
    const uint32_t zAddition = height % BLOCK_SIZE;
    auto xStepping = width/BLOCK_SIZE;
    auto zStepping = height/BLOCK_SIZE;

    auto pixelX = idX * xStepping + ((idX > xAddition) ? xAddition : idX);
    auto pixelZ = idZ * zStepping + ((idZ > zAddition) ? zAddition : idZ);

    xStepping = xStepping + ((xAddition <= 0) ? 0 : ((idX < xAddition) ? 1 : 0));
    zStepping = zStepping + ((zAddition <= 0) ? 0 : ((idZ < zAddition) ? 1 : 0));

    return Coordinates(pixelX, pixelZ, xStepping, zStepping);
}
}  // namespace

class Renderer
{
public:
    __device__ Renderer(const uint32_t samples, const uint32_t width, const uint32_t height, const uint8_t maxDepth,
        const Camera& camera)
        : camera_(camera)
        , height_(height)
        , samples_(samples)
        , width_(width)
        , maxDepth_(maxDepth)
        , objectsCount_(0)
    {}

    __device__ void setUp(ObjectData* objectsData, const uint32_t objectCount)
    {
        objectsCount_ = objectCount;
        objects_ = new AObject*[objectsCount_];
        for (uint32_t i=0; i<objectsCount_; i++)
        {
            if (objectsData[i].objectType_ == SphereData)
            {
                objects_[i] = new Sphere(objectsData[i].radius_, objectsData[i].position_, objectsData[i].emission_,
                    objectsData[i].color_, objectsData[i].reflectionType_);
            }
            else if (objectsData[i].objectType_ == PlaneData)
            {
                objects_[i] = new Plane(objectsData[i].north_, objectsData[i].east_, objectsData[i].position_,
                    objectsData[i].emission_, objectsData[i].color_, objectsData[i].reflectionType_);
            }
        }
    }

    __device__ void start(Vec3* image, const Vec3& vecZ)
    {
        const auto coordinates = calculateCoordinates(threadIdx.x, blockIdx.x, width_, height_);
        const auto limitZ = coordinates.zz_ + coordinates.loopZ_ + 1;
        const auto limitX = coordinates.xx_ + coordinates.loopX_ + 1;

        curandState state;
        auto seed = threadIdx.x + blockIdx.x * blockDim.x;
        curand_init(123456, seed, 0, &state);

        for (uint32_t z=coordinates.zz_; z<limitZ; z++)
        {
            for (uint32_t x=coordinates.xx_; x<limitX; x++)
            {
                const auto index = z * width_ + x;
                image[index] = samplePixel(camera_.orientation_, vecZ, x, z, samples_, state);
            }
        }
    }

private:
    __device__ Vec3 samplePixel(const containers::Vec3& vecX, const containers::Vec3& vecZ, const uint32_t pixelX,
    const uint32_t pixelZ, const uint32_t samples, curandState& state) const
    {
        const auto center = camera_.origin_;
        const auto direction = camera_.direction_;

        auto correctionX = (width_ % 2 == 0) ? 0.5 : 0.0;
        auto correctionZ = (width_ % 2 == 0) ? 0.5 : 0.0;
        double stepX = (pixelX < width_/2)
            ? width_/2 - pixelX - correctionX
            : ((double)width_/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
        double stepZ = (pixelZ < height_/2)
            ? height_/2 - pixelZ - correctionZ
            : ((double)height_/2 - pixelZ - 1.0) + ((correctionZ == 0.0) ? 1.0 : correctionZ);

        const auto gaze = (direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE).norm();

        Vec3 pixel = Vec3();
        for (uint32_t i=0; i<samples; i++)
        {
            // Tent filter
            const auto xFactor = tent_filter(state);
            const auto zFactor = tent_filter(state);
            const auto tentFilter = vecX * xFactor + vecZ * zFactor;
            // Tent filter

            const auto origin = center + vecX*stepX + vecZ*stepZ + tentFilter;
            pixel = pixel + firstLayer(Ray(origin + direction * VIEWPORT_DISTANCE, gaze), state);
        }

        pixel.xx_ = pixel.xx_/samples;
        pixel.yy_ = pixel.yy_/samples;
        pixel.zz_ = pixel.zz_/samples;

        return pixel;
    }

    __device__ Vec3 firstLayer(Ray ray, curandState& state) const
    {
        uint8_t depth = 0;
        const auto hitData = getHitObjectAndDistance(ray);
        if (hitData.index_ == -1) return Vec3();

        const auto& object = objects_[hitData.index_];
        const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
        const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

        depth++;
        Vec3 backData;
        if (depth < maxDepth_)
        {
            backData = secondLayer(reflected.ray_, depth, state) * reflected.power_;
            if (reflected.useSecond_)
            {
                backData = backData + secondLayer(reflected.secondRay_, depth, state) * reflected.secondPower_;
            }
        }

        return object->getEmission() + object->getColor().mult(backData);
    }

    __device__ Vec3 secondLayer(Ray ray, uint8_t& depth, curandState& state) const
    {
        const auto hitData = getHitObjectAndDistance(ray);
        if (hitData.index_ == -1) return Vec3();

        const auto& object = objects_[hitData.index_];
        const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
        const auto reflected = object->calculateReflections(intersection, ray.direction_, state, depth);

        depth++;
        Vec3 backData;
        if (depth < maxDepth_)
        {
            backData = deepLayers(reflected.ray_, depth, state) * reflected.power_;
            if (reflected.useSecond_)
            {
                backData = backData + deepLayers(reflected.secondRay_, depth, state) * reflected.secondPower_;
            }
        }

        return object->getEmission() + object->getColor().mult(backData);
    }

    __device__ Vec3 deepLayers(Ray ray, uint8_t depth, curandState& state) const
    {
        Vec3* objectEmissions = new Vec3[maxDepth_ - 2];
        Vec3* objectColors = new Vec3[maxDepth_ - 2];

        for (; depth<maxDepth_; depth++)
        {
            const auto hitData = getHitObjectAndDistance(ray);
            if (hitData.index_ == -1) break;

            const auto& object = objects_[hitData.index_];
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

    __device__ HitData getHitObjectAndDistance(const containers::Ray& ray) const
    {
        int index = -1;
        double distance = INF;

        for (uint32_t i=0; i<objectsCount_; i++)
        {
            auto temp = objects_[i]->intersect(ray);
            if (temp && temp < distance)
            {
                distance = temp;
                index = i;
            }
        }

        return HitData(index, distance);
    }

    AObject** objects_;
    Camera camera_;
    const uint32_t height_;
    const uint32_t samples_;
    const uint32_t width_;
    const uint8_t maxDepth_;
    uint32_t objectsCount_;
};

__global__ void cudaMain(Vec3* image, ObjectData* objectsData, const uint32_t objectsCount, const uint32_t width,
    const uint32_t height, Camera camera, Vec3 vecZ, uint32_t samples, const uint8_t maxDepth)
{
    Renderer render = Renderer(samples, width, height, maxDepth, camera);
    render.setUp(objectsData, objectsCount);
    render.start(image, vecZ);
}

}  // namespace tracer::render
