#include "renderer/RenderController.hpp"

#include <iostream>

#include "scene/objects/AObject.hpp"

#include "Renderer.cu"

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;

const uint32_t BLOCK_SIZE = 1024;
}  // namespace

RenderContoller::RenderContoller(SceneData& sceneData, const uint32_t samples)
    : sceneData_(sceneData)
    , samples_(samples)
{}

std::vector<containers::Vec3> RenderContoller::start()
{
    const auto imageSize = sceneData_.getHeight() * sceneData_.getWidth() * sizeof(Vec3);
    Vec3* devImage;
    cudaMalloc((void**)&devImage, imageSize);
    cudaMemset(devImage, 0, imageSize);

    auto objectDataVec = sceneData_.getObjectsData();
    AObject* devObjects[objectDataVec.size()];
    cudaMalloc(devObjects, sizeof(devObjects));

    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;
    cudaMain <<<numBlocks, numThreads>>> (devImage, devObjects, objectDataVec.data(), objectDataVec.size());

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(devImage);
    cudaFree(devObjects);

    const auto image = convertToVector(imagePtr);
    free(imagePtr);

    return image;
}

std::vector<containers::Vec3> RenderContoller::convertToVector(containers::Vec3* imagePtr)
{
    std::vector<Vec3> image;
    for (uint32_t iter = 0; iter < sceneData_.getHeight() * sceneData_.getWidth(); iter++)
    {
        image.push_back(imagePtr[iter]);
    }

    return image;
}

}  // namespace tracer::renderer
