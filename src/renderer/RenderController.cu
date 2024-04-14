#include "renderer/RenderController.hpp"

#include <iostream>

#include "objects/AObject.hpp"

#include "Constants.hpp"
#include "helpers/SceneConstants.hpp"
#include "helpers/PixelData.hpp"
#include "RendererCPU.hpp"
#include "RendererGPU.cu"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;

const float FOV_SCALE = 0.0009;

void cudaErrorCheck(const std::string& message)
{
    cudaError_t maybeError = cudaGetLastError();
    if (maybeError != cudaSuccess)
    {
        std::cout << message << " - error: " << cudaGetErrorString(maybeError) << std::endl;
    }
}

__global__ void cudaCreateObjects(AObject** objects, ObjectData* objectsData)
{
    if (objectsData[threadIdx.x].objectType_ == SphereData)
    {
        objects[threadIdx.x] = new Sphere(objectsData[threadIdx.x].radius_, objectsData[threadIdx.x].position_,
            objectsData[threadIdx.x].emission_, objectsData[threadIdx.x].color_,
            objectsData[threadIdx.x].reflectionType_);
    }
    else if (objectsData[threadIdx.x].objectType_ == PlaneData)
    {
        objects[threadIdx.x] = new Plane(objectsData[threadIdx.x].north_, objectsData[threadIdx.x].east_,
            objectsData[threadIdx.x].position_, objectsData[threadIdx.x].emission_, objectsData[threadIdx.x].color_,
            objectsData[threadIdx.x].reflectionType_);
    }
}

std::vector<uint32_t> prepareRandomSeends(const uint32_t ammount)
{
    std::vector<uint32_t> seeds;
    for (uint32_t i=0; i<ammount; i++)
    {
        seeds.push_back(rand());
    }

    return seeds;
}

}  // namespace

RenderController::RenderController(SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth)
    : camera_(sceneData.getCamera())
    , height_(sceneData.getHeight())
    , samples_(samples)
    , width_(sceneData.getWidth())
    , maxDepth_(maxDepth)
    , correctionX_((width_ % 2 == 0) ? 0.5 : 0.0)
    , correctionZ_((height_ % 2 == 0) ? 0.5 : 0.0)
    , image_(std::vector<Vec3> (width_ * height_))
{}

std::vector<containers::Vec3> RenderController::start(const std::vector<ObjectData>& objectDataVec)
{
    AObject** devObjects;
    cudaMalloc((void**)&devObjects, sizeof(AObject) * objectDataVec.size());
    cudaErrorCheck("Copy object blueprint data");

    ObjectData* devData;
    cudaMalloc((void**)&devData, sizeof(ObjectData) * objectDataVec.size());
    cudaMemcpy(devData, objectDataVec.data(), sizeof(ObjectData) * objectDataVec.size(), cudaMemcpyHostToDevice);
    cudaErrorCheck("Allocate memory for objects");

    cudaCreateObjects <<<1, objectDataVec.size()>>> (devObjects, devData);
    cudaErrorCheck("cudaCreateObjects kernel");

    cudaFree(devData);
    cudaErrorCheck("Clear object blueprint data");

    const Vec3 vecZ = (camera_.direction_%camera_.orientation_).norm();

    SceneConstants* devConstants;
    const auto constants = SceneConstants(camera_.orientation_, vecZ, camera_.origin_, camera_.direction_, samples_,
        maxDepth_, objectDataVec.size());
    cudaMalloc((void**)&devConstants, sizeof(SceneConstants));
    cudaMemcpy(devConstants, &constants, sizeof(SceneConstants), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy scene constants");

    const auto callback = [this, devObjects, devConstants, &vecZ](uint32_t z)
    {
        renderGPU(z, devObjects, devConstants, vecZ);
    };

    renderCPU(height_, callback);

    cudaDeviceReset();
    cudaErrorCheck("Reset device");

    return image_;
}

void RenderController::renderGPU(const uint32_t z, AObject** devObjects, SceneConstants* devConstants, const Vec3& vecZ)
{
    Vec3* devSamples;
    cudaMalloc((void**)&devSamples, sizeof(Vec3) * samples_);
    cudaErrorCheck("Set image array");

    PixelData* devPixelData;
    cudaMalloc((void**)&devPixelData, sizeof(PixelData));
    cudaErrorCheck("Allocate pixel data");

    Vec3* samplesPtr = (Vec3*)malloc(sizeof(Vec3) * samples_);

    // auto total = height_ * width_;
    for (uint32_t x = 0; x < 128/*width_*/; x++)
    {
        const auto index = z * width_ + x;
        image_[index] = startKernel(devObjects, devSamples, samplesPtr, devPixelData, devConstants, vecZ, x, z);
        // return;
    }

    cudaFree(devSamples);
    cudaFree(devPixelData);
    cudaErrorCheck("Free samples and pixel data arrays");

    free(samplesPtr);
}

Vec3 RenderController::startKernel(AObject** devObjects, Vec3* devSamples, Vec3* samplesPtr, PixelData* devPixelData,
    SceneConstants* devConstants, const Vec3& vecZ, const uint32_t pixelX, const uint32_t pixelZ)
{
    const auto vecX = camera_.orientation_;
    const auto direction = camera_.direction_;

    double stepX = (pixelX < width_/2)
        ? width_/2 - pixelX - correctionX_
        : ((double)width_/2 - pixelX - 1.0) + ((correctionX_ == 0.0) ? 1.0 : correctionX_);
    double stepZ = (pixelZ < height_/2)
        ? height_/2 - pixelZ - correctionZ_
        : ((double)height_/2 - pixelZ - 1.0) + ((correctionZ_ == 0.0) ? 1.0 : correctionZ_);

    const auto gaze = (direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE).norm();

    const auto pixelData = PixelData(stepX, stepZ, gaze);
    cudaMemcpy(devPixelData, &pixelData, sizeof(PixelData), cudaMemcpyHostToDevice);
    cudaErrorCheck("Allocate and copy pixel data");

    const auto numThreads = (samples_ <= THREAD_LIMIT) ? samples_ : THREAD_LIMIT;
    auto numBlocks = samples_ / numThreads + ((samples_ % numThreads) ? 1 : 0);
    numBlocks = (numBlocks <= BLOCK_LIMIT) ? numBlocks : BLOCK_LIMIT;

    uint32_t* devSeeds;
    cudaMalloc((void**)&devSeeds, sizeof(uint32_t) * numThreads * numBlocks);
    cudaMemcpy(devSeeds, prepareRandomSeends(numThreads * numBlocks).data(), sizeof(uint32_t) * numThreads * numBlocks,
        cudaMemcpyHostToDevice);
    cudaErrorCheck("Prepare random seeds");

    cudaMain <<<numBlocks, numThreads>>> (devSamples, devObjects, devConstants, devPixelData, devSeeds);
    cudaErrorCheck("Main kernel");

    cudaMemcpy(samplesPtr, devSamples, sizeof(Vec3) * samples_, cudaMemcpyDeviceToHost);
    cudaErrorCheck("Copy samples from device");

    auto pixel = Vec3();
    for (uint32_t i=0; i<samples_; i++)
    {
        pixel = pixel + samplesPtr[i];
    }

    cudaFree(devSeeds);

    pixel.xx_ = pixel.xx_/samples_;
    pixel.yy_ = pixel.yy_/samples_;
    pixel.zz_ = pixel.zz_/samples_;

    return pixel;
}

}  // namespace tracer::renderer
