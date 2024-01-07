#include "renderer/Renderer.hpp"

#include <omp.h>

//debug
#include <iostream>
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
    Vec3 stepX = Vec3(sceneData_.getWidth() * 0.5135 / sceneData_.getHeight());
    Vec3 stepY = (stepX%camera.direction_).norm() * 0.5135;
    // Also this should be able be const so idk why not worke worke
    // Another thing to check :b

    unsigned counter = 0;
    #pragma omp parallel for
    for (uint32_t y=0; y<sceneData_.getHeight(); y++)
    {
        fprintf(stdout, "\rRendering %g%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        for (uint32_t x=0; x<sceneData_.getWidth(); x++)
        {
            const auto index = y * sceneData_.getWidth() + x;
            image[index] = samplePixel(stepX, stepY, x, y);
        }
        counter++;
    }

    return image;
}

Vec3 Renderer::samplePixel(const Vec3 stepX, const Vec3 stepY, const uint32_t pixelX, const uint32_t pixelY)
{
    Vec3 pixel = Vec3();
    for (uint32_t i=0; i<samples_; i++)
    {
        // Here should be a filter that randomizes the ray send
        // So far I am not doing that as I don't exactly understand how this part works
        // That's why I'm trying to keep it as simple as possible

        Vec3 direction = stepX * ((/*0.25 + */pixelX)/sceneData_.getWidth() - 0.5)
            + stepY * ((/*0.25 + */pixelY)/sceneData_.getHeight() - 0.5)
            + sceneData_.getCamera().direction_;

        pixel = pixel + sendRay(Ray(sceneData_.getCamera().origin_ + direction * 140, direction.norm()), 0);

        // This whole thing is kinda ?? right now. So I need to work on this more to finally understand wtf is going on
        // here. Also it is posible that x and y are currently reversed <- this will require some work
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
