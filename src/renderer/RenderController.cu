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
}  // namespace

RenderContoller::RenderContoller(SceneData& sceneData, const uint32_t samples)
    : sceneData_(sceneData)
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

    auto objectDataVec = sceneData_.getObjectsData();
    AObject** devObjects = new AObject*[objectDataVec.size()];
    cudaMalloc(devObjects, sizeof(devObjects) * BLOCK_SIZE * BLOCK_SIZE);

    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;
    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    cudaMain <<<numBlocks, numThreads>>> (devImage, devObjects, objectDataVec.data(), objectDataVec.size(),
        sceneData_.getWidth(), sceneData_.getHeight(), camera, vecZ, samples_);

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaDeviceReset();

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
