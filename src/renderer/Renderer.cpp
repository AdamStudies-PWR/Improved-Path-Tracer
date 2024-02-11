#include "renderer/Renderer.hpp"

#include <omp.h>

//debug
#include <iostream>
#include <fstream>
//

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;

const uint16_t VIEWPORT_DISTANCE = 140;
const uint8_t MAX_DEPTH = 10;
const float FOV_SCALE = 0.0009;

std::uniform_real_distribution<> tent_filter(-1.0, 1.0);
}  // namespace

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
    std::vector<Vec3> image (sceneData_.getWidth() * sceneData_.getHeight());

    // Check why I need to copy and store the camera first?
    auto camera = sceneData_.getCamera();
    const Vec3 vecZ = (camera.direction_%camera.orientation_).norm();

    unsigned counter = 0;
    #pragma omp parallel for
    for (uint32_t z=0; z<sceneData_.getHeight(); z++)
    {
        fprintf(stdout, "\rRendering %.2f%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        #pragma omp parallel for
        for (uint32_t x=0; x<sceneData_.getWidth(); x++)
        {
            const auto index = z * sceneData_.getWidth() + x;
            image[index] = samplePixel(camera.orientation_, vecZ, x, z);
        }
        counter++;
    }

    return image;
}

Vec3 Renderer::samplePixel(const Vec3& vecX, const Vec3& vecZ, const uint32_t pixelX, const uint32_t pixelZ)
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
