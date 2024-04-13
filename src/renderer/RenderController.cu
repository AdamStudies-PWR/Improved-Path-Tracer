#include "renderer/RenderController.hpp"

#include <iostream>

#include "objects/AObject.hpp"

#include "Constants.hpp"
#include "helpers/SceneConstants.hpp"
#include "RendererCPU.hpp"
#include "RendererGPU.cu"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;

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
    : sceneData_(sceneData)
    , maxDepth_(maxDepth)
    , samples_(samples)
    , image_(std::vector<Vec3> (sceneData_.getWidth() * sceneData_.getHeight()))
{}

std::vector<containers::Vec3> RenderController::start()
{
    auto objectDataVec = sceneData_.getObjectsData();

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

    auto camera = sceneData_.getCamera();
    const Vec3 vecZ = (camera.direction_%camera.orientation_).norm();

    SceneConstants* devConstants;
    const auto constants = SceneConstants(camera.orientation_, vecZ, camera.origin_, camera.direction_, samples_,
        maxDepth_, objectDataVec.size(), sceneData_.getWidth(), sceneData_.getHeight());
    cudaMalloc((void**)&devConstants, sizeof(SceneConstants));
    cudaMemcpy(devConstants, &constants, sizeof(SceneConstants), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy scene constants");

    const auto callback = [this, devObjects, devConstants, &vecZ](uint32_t z)
    {
        renderGPU(z, devObjects, devConstants);
    };

    renderCPU(sceneData_.getHeight(), callback);

    cudaDeviceReset();
    cudaErrorCheck("Reset device");

    return image_;
}

void RenderController::renderGPU(const uint32_t z, AObject** devObjects, SceneConstants* devConstants)
{
    Vec3* devRow;
    cudaMalloc((void**)&devRow, sizeof(Vec3) * sceneData_.getWidth());
    cudaMemset(devRow, 0, sizeof(Vec3) * sceneData_.getWidth());
    cudaErrorCheck("Allocate row data");

    startKernel(devRow, devObjects, devConstants, z);

    cudaFree(devRow);
    cudaErrorCheck("Free allocated data");
}

void RenderController::startKernel(Vec3* devRow, AObject** devObjects, SceneConstants* devConstants, const uint32_t z)
{
    const auto numThreads = (sceneData_.getWidth() <= THREAD_LIMIT) ? sceneData_.getWidth() : THREAD_LIMIT;
    auto numBlocks = sceneData_.getWidth() / numThreads + ((sceneData_.getWidth() % numThreads > 0) ? 1 : 0);
    numBlocks = (numBlocks <= BLOCK_LIMIT) ? numBlocks : BLOCK_LIMIT;

    uint32_t* devSeeds;
    cudaMalloc((void**)&devSeeds, sizeof(uint32_t) * numThreads * numBlocks);
    cudaMemcpy(devSeeds, prepareRandomSeends(numThreads * numBlocks).data(), sizeof(uint32_t) * numThreads * numBlocks,
        cudaMemcpyHostToDevice);
    cudaErrorCheck("Prepare random seeds");

    cudaMain <<<numBlocks, numThreads>>> (devRow, devObjects, devConstants, devSeeds, z);
    cudaErrorCheck("Main kernel");

    Vec3* rowPtr = (Vec3*)malloc(sizeof(Vec3) * sceneData_.getWidth());
    cudaMemcpy(rowPtr, devRow, sizeof(Vec3) * sceneData_.getWidth(), cudaMemcpyDeviceToHost);
    cudaErrorCheck("Copy row from device");

    for (uint32_t i=0; i<sceneData_.getWidth(); i++)
    {
        const auto index = z * sceneData_.getWidth() + i;
        image_[index] = rowPtr[i];
    }

    free(rowPtr);
}

}  // namespace tracer::renderer
