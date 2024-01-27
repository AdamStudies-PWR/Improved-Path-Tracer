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

// const uint16_t MAX_DEPTH = 10;
// For now no reflections
const uint16_t VIEW_PORT_DISTANCE = 140;
const uint16_t MAX_DEPTH = 0;
const float FOV_SCALE = 0.001;

std::uniform_real_distribution<> zero_to_hunderd_distribution(0.0, 100.0);
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
        fprintf(stdout, "\rRendering %g%%", (counter * 100.)/(sceneData_.getHeight() - 1));
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

    Vec3 pixel = Vec3();

    for (uint32_t i=0; i<samples_; i++)
    //for (uint32_t i=0; i<1; i++)
    {
        // tutaj powinno dojść losowanie - biorę próbki w obszarze wokół pixela +1 -1

        const auto origin = center + vecX*stepX + vecZ*stepZ;
        const auto gaze = direction + vecX*stepX*FOV_SCALE + vecZ*stepZ*FOV_SCALE;
        pixel = pixel + sendRay(Ray(origin + direction * VIEW_PORT_DISTANCE, gaze), 0);
    }

    pixel.xx_ = pixel.xx_/samples_;
    pixel.yy_ = pixel.yy_/samples_;
    pixel.zz_ = pixel.zz_/samples_;

    return pixel;
}

Vec3 Renderer::sendRay(const Ray& ray, uint16_t depth)
{
    if (depth > MAX_DEPTH) return Vec3();

    const auto pair = sceneData_.getHitObjectAndDistance(ray);
    if (pair.first == -1) return Vec3();

    const auto& object = sceneData_.getObjectAt(pair.first);

    // debug switch
    /*switch (object->getReflectionType())
    {
        case objects::EReflectionType::Diffuse: return Vec3(1, 0, 0);
        case objects::EReflectionType::Refractive: return Vec3(0, 1, 0);
        case objects::EReflectionType::Specular: return Vec3(0, 0, 1);
        default: return Vec3();
    }*/

    return object->getColor();

    // This will show what objects are hit so hopefully we will get some usefull information

    /*const auto intersection = ray.origin_ + (ray.direction_*pair.second);

    const auto reflectedRay = object->getReflectedRay(ray, intersection);

    return object->getEmission() + object->getColor().mult(sendRay(reflectedRay, ++depth));*/
}

}  // namespace tracer::renderer
