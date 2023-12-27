#include "renderer/Renderer2.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;

std::uniform_real_distribution<> zero_to_hunderd_distribution(0.0, 100.0);
}  // namespace

Renderer2::Renderer2(SceneData& sceneData, const uint32_t samples)
    : samples_(samples)
    , sceneData_(sceneData)
{
    std::random_device rd;
    std::mt19937 generator(rd());
    generator_ = generator;
}

std::vector<Vec3> Renderer2::render()
{
    std::vector<Vec3> image {};

    for (uint32_t i=0; i<sceneData_.getHeight(); i++)
    {
        for (uint32_t j=0; j<sceneData_.getWidth(); j++)
        {
            image.push_back(sendRay());
        }
    }

    return image;
}

Vec3 Renderer2::sendRay()
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
