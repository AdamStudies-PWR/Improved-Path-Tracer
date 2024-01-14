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
const uint16_t MAX_DEPTH = 0;

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
    // Those two are kinda magic for me so I need to get explanation for why is it calculated like that
    // Vec3 stepX = Vec3(sceneData_.getWidth() * 0.5135 / sceneData_.getHeight());
    // Vec3 stepY = (stepX%camera.direction_).norm() * 0.5135;
    // Also this should be able be const so idk why not worke worke
    // Another thing to check :b

    //camera.orientation_ = Vec3(((camera.direction_.yy_ - camera.direction_.zz_) / camera.direction_.yy_), 1, 1).norm();
    Vec3 vecY = (camera.direction_%camera.orientation_).norm();

    unsigned counter = 0;
    #pragma omp parallel for
    for (uint32_t y=0; y<sceneData_.getHeight(); y++)
    {
        fprintf(stdout, "\rRendering %g%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        for (uint32_t x=0; x<sceneData_.getWidth(); x++)
        {
            const auto index = y * sceneData_.getWidth() + x;
            image[index] = samplePixel(camera.orientation_, vecY, x, y);
        }
        counter++;
    }

    return image;
}

Vec3 Renderer::samplePixel(Vec3 vecX, Vec3 vecY, uint32_t pixelX, uint32_t pixelY)
{
    const auto center = sceneData_.getCamera().origin_;

    auto correctionX = (sceneData_.getWidth() % 2 == 0) ? 0.5 : 0.0;
    auto correctionY = (sceneData_.getWidth() % 2 == 0) ? 0.5 : 0.0;
    double stepX = (pixelX < sceneData_.getWidth()/2)
        ? sceneData_.getWidth()/2 - pixelX - correctionX
        : ((double)sceneData_.getWidth()/2 - pixelX - 1.0) + ((correctionX == 0.0) ? 1.0 : correctionX);
    double stepY = (pixelY < sceneData_.getHeight()/2)
        ? sceneData_.getHeight()/2 - pixelY - correctionY
        : ((double)sceneData_.getHeight()/2 - pixelY - 1.0) + ((correctionY == 0.0) ? 1.0 : correctionY);

    Vec3 pixel = Vec3();

    for (uint32_t i=0; i<samples_; i++)
    //for (uint32_t i=0; i<1; i++)
    {
        auto origin = center + vecX*stepX + vecY*stepY;
        /*Vec3 direction = vecX * ((0.25 + pixelX)/sceneData_.getWidth() - 0.5)
            + vecY * ((0.25 + pixelY)/sceneData_.getHeight() - 0.5)
            + sceneData_.getCamera().direction_;

        pixel = pixel + sendRay(Ray(sceneData_.getCamera().origin_ + direction * 140, direction.norm() * -1), 0);*/
        const auto ray = Ray(origin + sceneData_.getCamera().direction_ * 140, sceneData_.getCamera().direction_);
        pixel = pixel + sendRay(ray, 0);
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
