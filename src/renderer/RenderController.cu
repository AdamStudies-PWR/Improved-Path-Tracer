#include "renderer/RenderController.hpp"

#include <iostream>

#include "objects/AObject.hpp"

#include "Constants.hpp"
#include "Renderer.cu"


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
        std::cout << message << " error: " << cudaGetErrorString(maybeError) << std::endl;
    }
}
}  // namespace

RenderContoller::RenderContoller(SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth)
    : sceneData_(sceneData)
    , maxDepth_(maxDepth)
    , samples_(samples)
{}

std::vector<containers::Vec3> RenderContoller::start()
{
    auto camera = sceneData_.getCamera();
    const Vec3 vecZ = (camera.direction_%camera.orientation_).norm();

    const auto imageSize = sceneData_.getHeight() * sceneData_.getWidth() * sizeof(Vec3);
    Vec3* devImage;
    cudaMalloc((void**)&devImage, imageSize);
    cudaMemset(devImage, 0, imageSize);
    cudaErrorCheck("Set image array");

    auto objectDataVec = sceneData_.getObjectsData();
    ObjectData* devData;
    cudaMalloc((void**)&devData, sizeof(ObjectData) * objectDataVec.size());
    cudaMemcpy(devData, objectDataVec.data(), sizeof(ObjectData) * objectDataVec.size(), cudaMemcpyHostToDevice);
    cudaErrorCheck("Set object data array");

    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;
    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    cudaMain <<<numBlocks, numThreads>>> (devImage, devData, objectDataVec.size(), sceneData_.getWidth(),
        sceneData_.getHeight(), camera, vecZ, samples_, maxDepth_);
    cudaErrorCheck("cudaMain kernel");

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaErrorCheck("Copy to device");

    cudaDeviceReset();
    cudaErrorCheck("Reset device");

    const auto image = convertToVector(imagePtr);
    free(imagePtr);

    return image;
}

std::vector<Vec3> RenderContoller::convertToVector(Vec3* imagePtr)
{
    std::vector<Vec3> image;
    for (uint32_t iter = 0; iter < sceneData_.getHeight() * sceneData_.getWidth(); iter++)
    {
        image.push_back(imagePtr[iter]);
    }

    return image;
}

}  // namespace tracer::renderer
