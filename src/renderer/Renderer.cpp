#include "renderer/Renderer.hpp"

#include <omp.h>

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;

const uint16_t MAX_DEPTH = 10;

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
    Vec3 stepX = Vec3((sceneData_.getWidth() * 0.5135)/sceneData_.getHeight());
    Vec3 stepY = (stepX%camera.direction_).norm() * 0.5135;
    // Also this should be able be const so idk why not worke worke
    // Another thing to check :b

    unsigned counter = 0;
    #pragma omp parallel for
    for (uint32_t i=0; i<sceneData_.getHeight(); i++)
    {
        fprintf(stdout, "\rRendering %g%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        for (uint32_t j=0; j<sceneData_.getWidth(); j++)
        {
            const auto index = i * sceneData_.getWidth() + j;
            image[index] = samplePixel(stepX, stepY, i, j);
        }
        counter++;
    }

    return image;
}

Vec3 Renderer::samplePixel(const Vec3 stepX, const Vec3 stepY, const uint32_t pixelX, const uint32_t pixelY)
{
    Vec3 pixel = Vec3(0, 0, 0);
    for (uint32_t i=0; i<samples_; i++)
    {
        // Here should be a filter that randomizes the ray send
        // So far I am not doing that as I don't exactly understand how this part works
        // That's why I'm trying to keep it as simple as possible

        Vec3 origin = stepX * (pixelX/sceneData_.getHeight())
            + stepY * (pixelY/sceneData_.getWidth())
            + sceneData_.getCamera().direction_;

        pixel = pixel + sendRay(Ray(sceneData_.getCamera().origin_ + origin * 140, origin.norm()), 0);

        // This whole thing is kinds ?? right now. So I need to work on this more to finally understand wtf is going on
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
    const auto intersection = ray.origin_ + (ray.direction_*pair.second);

    const auto reflectedRay = object->getReflectedRay(ray, intersection);

    return sendRay(reflectedRay, ++depth);
}

}  // namespace tracer::renderer
