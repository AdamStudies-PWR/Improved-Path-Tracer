#include "renderer/Renderer.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;

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
    Vec3 ray;

    unsigned counter = 0;
    #pragma omp parallel for private(ray)
    for (uint32_t i=0; i<sceneData_.getHeight(); i++)
    {
        fprintf(stdout, "\nRendering %g%%", (counter * 100.)/(sceneData_.getHeight() - 1));
        for (uint32_t j=0; j<sceneData_.getWidth(); j++)
        {
            const auto index = i * sceneData_.getWidth() + j;
            image[index] = sendRay();
        }
        counter++;
    }

    return image;
}

Vec3 Renderer::sendRay()
{
    Vec3 pixel = Vec3(0, 0, 0);
    for (uint32_t i=0; i<samples_; i++)
    {
        pixel.xx_ = pixel.xx_ + zero_to_hunderd_distribution(generator_);
        pixel.yy_ = pixel.yy_ + zero_to_hunderd_distribution(generator_);
        pixel.zz_ = pixel.zz_ + zero_to_hunderd_distribution(generator_);
    }

    pixel.xx_ = pixel.xx_/samples_;
    pixel.yy_ = pixel.yy_/samples_;
    pixel.zz_ = pixel.zz_/samples_;

    return pixel;
}

}  // namespace tracer::renderer
