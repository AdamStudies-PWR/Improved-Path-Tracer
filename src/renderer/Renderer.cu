#include "renderer/Renderer.hpp"

#include <iostream>

#include <curand_kernel.h>

#include "scene/objects/AObject.hpp"
#include "scene/objects/Plane.cu"
#include "scene/objects/Sphere.cu"
#include "utils/CudaUtils.hpp"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene::objects;
using namespace scene;
using namespace utils;

__device__ const float FOV_SCALE = 0.0009;
__device__ const uint16_t VIEWPORT_DISTANCE = 140;
__device__ const uint32_t BLOCK_SIZE = 1024;
__device__ const uint8_t MAX_DEPTH = 10;

__device__ void createObjects(AObject** objects, ObjectData** data, const uint32_t objectCount)
{
    for (uint32_t i=0; i<objectCount; i++)
    {
        if (data[i])
        {
            printf("Pain");
            double radius = data[i]->position_.xx_;
            printf("Oki doki%f\n", radius);
            // printf("Type: %f", data[i]->position_.xx_);
        }
        else
        {
            printf("Nok\n");
        }

        if (data[i]->objectType_ == "Sphere")
        {
            printf("Wchodzę tutaj!");
            // printf("Type: %s", data[i]->objectType_);
            // auto* sphere = new Sphere(data[i]->radius_, data[i]->position_, data[i]->emission_, data[i]->color_,
            //     data[i]->reflectionType_);
            //objects[i] = sphere;
        }
        else if (data[i]->objectType_ == "Plane")
        {
            printf("Wchodzę tutaj2!");
            //objects[i] = new Plane(data[i]->north_, data[i]->east_, data[i]->position_, data[i]->emission_,
                //data[i]->color_, data[i]->reflectionType_);
        }
        else
        {
            printf("Wchodzę tutaj3!");
        }
    }
}

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

__device__ Vec3 sendRay(const Ray& ray, uint8_t depth, SceneData& data, curandState& state)
{
    if (depth > MAX_DEPTH) return Vec3();

    const auto hitData = data.getHitObjectAndDistance(ray);
    // if (hitData.index_ == -1) return Vec3();

    // const auto* object = data.getObjectAt(hitData.distance_);

    /*Some stopping condition based on reflectivness - should be random*/
    /*Skipping for now*/

    // return object->getColor();

    /*const auto intersection = ray.origin_ + ray.direction_ * hitData.distance_;
    const auto wrappedRays = object->calculateReflections(intersection, ray.direction_, state, depth);

    if (wrappedRays.size_ <= 0)
    {
        return Vec3();
    }

    Vec3 path = Vec3();
    ++depth;
    for (int i=0; i<wrappedRays.size_; i++)
    {
        path = path + sendRay(wrappedRays.data_[i].ray_, depth, data, state) * wrappedRays.data_[i].power_;
    }

    return object->getEmission() + object->getColor().mult(path);*/
}

__device__ Vec3 samplePixel(const containers::Vec3& vecX, const containers::Vec3& vecZ, const uint32_t pixelX,
    const uint32_t pixelZ, SceneData& data, const uint32_t samples, curandState& state)
{
    const auto center = data.getCamera().origin_;
    const auto direction = data.getCamera().direction_;

    auto correctionX = (data.getWidth() % 2 == 0) ? 0.5 : 0.0;
    auto correctionZ = (data.getWidth() % 2 == 0) ? 0.5 : 0.0;
    double stepX = (pixelX < data.getWidth()/2)
        ? data.getWidth()/2 - pixelX - correctionX
        : ((double)data.getWidth()/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
    double stepZ = (pixelZ < data.getHeight()/2)
        ? data.getHeight()/2 - pixelZ - correctionZ
        : ((double)data.getHeight()/2 - pixelZ - 1.0) + ((correctionZ == 0.0) ? 1.0 : correctionZ);

    const auto gaze = direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE;

    Vec3 pixel = Vec3();
    for (uint32_t i=0; i<samples; i++)
    {
        // Tent filter
        const auto xFactor = tent_filter(state);
        const auto zFactor = tent_filter(state);
        const auto tentFilter = vecX * xFactor + vecZ * zFactor;
        // Tent filter

        const auto origin = center + vecX*stepX + vecZ*stepZ + tentFilter;
        pixel = pixel + sendRay(Ray(origin + direction * VIEWPORT_DISTANCE, gaze), 0, data, state);
    }

    pixel.xx_ = pixel.xx_/samples;
    pixel.yy_ = pixel.yy_/samples;
    pixel.zz_ = pixel.zz_/samples;

    return pixel;
}

__global__ void cudaRender(Vec3* image, AObject** objects, const Vec3 vecZ, SceneData data, uint32_t samples)
{
    // printf("Test\n");
    createObjects(objects, data.getObjectsData(), data.getObjectCount());

    const auto coordinates = calculateCoordinates(threadIdx.x, blockIdx.x, data.getWidth(), data.getHeight());
    const auto limitZ = coordinates.zz_ + coordinates.loopZ_ + 1;
    const auto limitX = coordinates.xx_ + coordinates.loopX_ + 1;

    curandState state;
    auto seed = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(123456, seed, 0, &state);

    for (uint32_t z=coordinates.zz_; z<limitZ; z++)
    {
        for (uint32_t x=coordinates.xx_; x<limitX; x++)
        {
            const auto index = z * data.getWidth() + x;
            image[index] = samplePixel(data.getCamera().orientation_, vecZ, x, z, data, samples, state);
        }
    }
}

}  // namespace

__device__ Coordinates::Coordinates(uint32_t xx, uint32_t zz, uint32_t loopX, uint32_t loopZ)
    : xx_(xx)
    , zz_(zz)
    , loopX_(loopX)
    , loopZ_(loopZ)
{}

Renderer::Renderer(SceneData& sceneData, const uint32_t samples)
    : samples_(samples)
    , sceneData_(sceneData)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    generator_ = generator;
}

std::vector<Vec3> Renderer::render()
{
    auto camera = sceneData_.getCamera();
    const Vec3 vecZ = (camera.direction_%camera.orientation_).norm();

    const auto imageSize = sceneData_.getHeight() * sceneData_.getWidth() * sizeof(Vec3);
    Vec3* devImage;
    cudaMalloc((void**)&devImage, imageSize);
    cudaMemset(devImage, 0, imageSize);

    AObject* devObjects[sceneData_.getObjectCount()];
    cudaMalloc(devObjects, sizeof(devObjects));

    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;

    cudaRender <<<numBlocks, numThreads>>> (devImage, devObjects, vecZ, sceneData_, samples_);

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(devImage);
    cudaFree(devObjects);

    std::vector<Vec3> image;
    for (uint32_t iter = 0; iter < sceneData_.getHeight() * sceneData_.getWidth(); iter++)
    {
        image.push_back(imagePtr[iter]);
    }

    free(imagePtr);
    return image;
}

Vec3 Renderer::samplePixel2(const Vec3& vecX, const Vec3& vecZ, const uint32_t pixelX, const uint32_t pixelZ)
{
    /*const auto center = sceneData_.getCamera().origin_;
    const auto direction = sceneData_.getCamera().direction_;

    auto correctionX = (sceneData_.getWidth() % 2 == 0) ? 0.5 : 0.0;
    auto correctionZ = (sceneData_.getWidth() % 2 == 0) ? 0.5 : 0.0;
    double stepX = (pixelX < sceneData_.getWidth()/2)
        ? sceneData_.getWidth()/2 - pixelX - correctionX
        : ((double)sceneData_.getWidth()/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
    double stepZ = (pixelZ < sceneData_.getHeight()/2)
        ? sceneData_.getHeight()/2 - pixelZ - correctionZ
        : ((double)sceneData_.getHeight()/2 - pixelZ - 1.0) + ((correctionZ == 0.0) ? 1.0 : correctionZ);

    const auto gaze = direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE;

    Vec3 pixel = Vec3();
    for (uint32_t i=0; i<samples_; i++)
    {
        // Tent filter
        const auto xFactor = tent_filter(generator_);
        const auto zFactor = tent_filter(generator_);
        const auto tentFilter = vecX * xFactor + vecZ * zFactor;
        // Tent filter

        const auto origin = center + vecX*stepX + vecZ*stepZ + tentFilter;
        pixel = pixel + sendRay2(Ray(origin + direction * VIEWPORT_DISTANCE, gaze), 0);
    }

    pixel.xx_ = pixel.xx_/samples_;
    pixel.yy_ = pixel.yy_/samples_;
    pixel.zz_ = pixel.zz_/samples_;

    return pixel;*/
    return Vec3();
}

Vec3 Renderer::sendRay2(const Ray& ray, uint8_t depth)
{
    /*if (depth > MAX_DEPTH) return Vec3();

    const auto hitData = sceneData_.getHitObjectAndDistance(ray);
    if (hitData.first == -1) return Vec3();

    const auto& object = sceneData_.getObjectAt(hitData.first);*/

    /*Some stopping condition based on reflectivness - should be random*/
    /*Skipping for now*/

    /*const auto intersection = ray.origin_ + ray.direction_ * hitData.second;
    const auto reflectedRays = object->calculateReflections(intersection, ray.direction_, generator_, depth);

    Vec3 path = Vec3();
    ++depth;
    for (const auto& reflectionData : reflectedRays)
    {
        path = path + sendRay2(reflectionData.first, depth) * reflectionData.second;
    }

    return object->getEmission() + object->getColor().mult(path);*/
    return Vec3();
}

}  // namespace tracer::renderer
