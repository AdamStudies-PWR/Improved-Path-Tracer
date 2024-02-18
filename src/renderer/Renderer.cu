#include "renderer/Renderer.hpp"

#include <iostream>

// #include "utils/CudaUtils.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;

const uint16_t VIEWPORT_DISTANCE = 140;
const uint8_t MAX_DEPTH = 10;
const float FOV_SCALE = 0.0009;

__device__ const uint32_t BLOCK_SIZE = 1024;

std::uniform_real_distribution<> tent_filter(-1.0, 1.0);

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

__global__ void samplePixel(Vec3* image, const uint32_t width, const uint32_t height)
{
    const auto coordinates = calculateCoordinates(threadIdx.x, blockIdx.x, width, height);
    const auto limitZ = coordinates.zz_ + coordinates.loopZ_ + 1;
    const auto limitX = coordinates.xx_ + coordinates.loopX_ + 1;

    for (uint32_t z=coordinates.zz_; z<limitZ; z++)
    {
        for (uint32_t x=coordinates.xx_; x<limitX; x++)
        {
            const auto index = z * width + x;
            image[index].xx_ = 255;
            image[index].yy_ = 255;
            image[index].zz_ = 255;
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

    const auto numBlocks = (sceneData_.getHeight() <= BLOCK_SIZE) ? sceneData_.getHeight() : BLOCK_SIZE;
    const auto numThreads = (sceneData_.getWidth() <= BLOCK_SIZE) ? sceneData_.getWidth() : BLOCK_SIZE;

    samplePixel <<<numBlocks, 1024>>> (devImage, sceneData_.getWidth(), sceneData_.getHeight());

    Vec3* imagePtr = (Vec3*)malloc(imageSize);
    cudaMemcpy(imagePtr, devImage, imageSize, cudaMemcpyDeviceToHost);
    cudaFree(devImage);

    std::vector<Vec3> image;
    for (uint32_t iter = 0; iter < sceneData_.getHeight() * sceneData_.getWidth(); iter++)
    {
        //if (imagePtr[iter].xx_ != 0.0) std::cout << "It worked! " << std::endl;
        image.push_back(imagePtr[iter]);
    }

    /*unsigned counter = 0;
    for (uint32_t z=0; z<sceneData_.getHeight(); z++)
    {
        fprintf(stdout, "\rRendering %.2f%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        for (uint32_t x=0; x<sceneData_.getWidth(); x++)
        {
            const auto index = z * sceneData_.getWidth() + x;
            image[index] = samplePixel(camera.orientation_, vecZ, x, z);
        }
        counter++;
    }

    return image;*/

    free(imagePtr);
    return image;
}

Vec3 Renderer::samplePixel2(const Vec3& vecX, const Vec3& vecZ, const uint32_t pixelX, const uint32_t pixelZ)
{
    const auto center = sceneData_.getCamera().origin_;
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
        pixel = pixel + sendRay(Ray(origin + direction * VIEWPORT_DISTANCE, gaze), 0);
    }

    pixel.xx_ = pixel.xx_/samples_;
    pixel.yy_ = pixel.yy_/samples_;
    pixel.zz_ = pixel.zz_/samples_;

    return pixel;
}

Vec3 Renderer::sendRay(const Ray& ray, uint8_t depth)
{
    if (depth > MAX_DEPTH) return Vec3();

    const auto hitData = sceneData_.getHitObjectAndDistance(ray);
    if (hitData.first == -1) return Vec3();

    const auto& object = sceneData_.getObjectAt(hitData.first);

    /*Some stopping condition based on reflectivness - should be random*/
    /*Skipping for now*/

    const auto intersection = ray.origin_ + ray.direction_ * hitData.second;
    const auto reflectedRays = object->calculateReflections(intersection, ray.direction_, generator_, depth);

    Vec3 path = Vec3();
    ++depth;
    for (const auto& reflectionData : reflectedRays)
    {
        path = path + sendRay(reflectionData.first, depth) * reflectionData.second;
    }

    return object->getEmission() + object->getColor().mult(path);
}

}  // namespace tracer::renderer
