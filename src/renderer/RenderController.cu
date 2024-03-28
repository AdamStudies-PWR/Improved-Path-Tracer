#include "renderer/RenderController.hpp"

#include <iostream>

#include "objects/AObject.hpp"

#include "Constants.hpp"
#include "Renderer.cu"
#include "ImageData.hpp"


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
}  // namespace

RenderContoller::RenderContoller(SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth)
    : sceneData_(sceneData)
    , maxDepth_(maxDepth)
    , samples_(samples)
{}

std::vector<containers::Vec3> RenderContoller::start()
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
    Camera* devCamera;
    cudaMalloc((void**)&devCamera, sizeof(Camera));
    cudaMemcpy(devCamera, &camera, sizeof(Camera), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy camera data");

    Vec3* devVecZ;
    cudaMalloc((void**)&devVecZ, sizeof(Vec3));
    cudaMemcpy(devVecZ, &(camera.direction_%camera.orientation_).norm(), sizeof(Vec3), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy vecZ data");

    ImageData* devImageData;
    const auto imageProperties = ImageData(sceneData_.getWidth(), sceneData_.getHeight(), samples_, maxDepth_,
        objectDataVec.size());
    cudaMalloc((void**)&devImageData, sizeof(ImageData));
    cudaMemcpy(devImageData, &imageProperties, sizeof(ImageData), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy image properties data");

    const auto imageSize = sceneData_.getHeight() * sceneData_.getWidth() * sizeof(Vec3);
    Vec3* devImage;
    cudaMalloc((void**)&devImage, imageSize);
    cudaMemset(devImage, 0, imageSize);
    cudaErrorCheck("Set image array");

    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;
    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    cudaMain <<<numBlocks, numThreads>>> (devImage, devObjects, devCamera, devVecZ, devImageData);
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
