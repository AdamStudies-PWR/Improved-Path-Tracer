#include "renderer/RenderController.hpp"

#include <iostream>

#include "objects/AObject.hpp"

#include "Constants.hpp"
#include "helpers/ImageData.hpp"
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
    auto propsDataVec = sceneData_.getPropsData();
    auto lightsDataVec = sceneData_.getLightsData();
    const auto propsSize = sizeof(AObject) * propsDataVec.size();
    const auto lightsSize = sizeof(AObject) * lightsDataVec.size();

    ObjectData* devPropsData;
    cudaMalloc((void**)&devPropsData, sizeof(ObjectData) * propsDataVec.size());
    cudaMemcpy(devPropsData, propsDataVec.data(), sizeof(ObjectData) * propsDataVec.size(), cudaMemcpyHostToDevice);
    cudaErrorCheck("Allocate memory for props data");

    AObject** devProps;
    cudaMalloc((void**)&devProps, propsSize);
    cudaErrorCheck("Allocate props data");

    cudaCreateObjects <<<1, propsDataVec.size()>>> (devProps, devPropsData);
    cudaErrorCheck("cudaCreateObjects kernel");

    cudaFree(devPropsData);
    cudaErrorCheck("Clear props blueprint data");

    ObjectData* devLightsData;
    cudaMalloc((void**)&devLightsData, sizeof(ObjectData) * lightsDataVec.size());
    cudaMemcpy(devLightsData, lightsDataVec.data(), sizeof(ObjectData) * lightsDataVec.size(), cudaMemcpyHostToDevice);
    cudaErrorCheck("Allocate memory for lights data");

    AObject** devLights;
    cudaMalloc((void**)&devLights, lightsSize);
    cudaErrorCheck("Allocate light sources data");

    cudaCreateObjects <<<1, lightsDataVec.size()>>> (devLights, devLightsData);
    cudaErrorCheck("cudaCreateObjects kernel");

    cudaFree(devLightsData);
    cudaErrorCheck("Clear lights blueprint data");

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
        propsDataVec.size(), lightsDataVec.size());
    cudaMalloc((void**)&devImageData, sizeof(ImageData));
    cudaMemcpy(devImageData, &imageProperties, sizeof(ImageData), cudaMemcpyHostToDevice);
    cudaErrorCheck("Copy image properties data");

    const auto imageSize = sceneData_.getHeight() * sceneData_.getWidth() * sizeof(Vec3);
    Vec3* devImage;
    cudaMalloc((void**)&devImage, imageSize);
    cudaMemset(devImage, 0, imageSize);
    cudaErrorCheck("Set image array");

    const auto numThreads = (sceneData_.getWidth() <= THREAD_SIZE) ? sceneData_.getWidth() : THREAD_SIZE;
    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;

    cudaMain <<</*1, 1*/numBlocks, numThreads>>> (devImage, devProps, devLights, devCamera, devVecZ, devImageData);
    cudaErrorCheck("cudaMain kernel");

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaErrorCheck("Copy from device");

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
