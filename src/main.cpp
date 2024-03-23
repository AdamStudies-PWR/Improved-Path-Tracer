#include <functional>
#include <iostream>
#include <memory>

#include "containers/Vec3.hpp"
#include "renderer/Renderer.hpp"
#include "scene/SceneData.hpp"
#include "utils/Image.hpp"
#include "utils/Measurements.hpp"
#include "utils/InputParser.hpp"

namespace
{
using namespace tracer::containers;
using namespace tracer::renderer;
using namespace tracer::scene;
using namespace tracer::utils;
}  // namespace

int main(int argc, char* argv[])
{
    InputParser inputParser((argc - 1), argv);
    if (not inputParser.isInputValid())
    {
        return 0;
    }

    SceneData sceneData(inputParser.getScenePath());
    if (not sceneData.initScene())
    {
        return 0;
    }

    auto renderer = std::make_shared<Renderer>(sceneData, inputParser.getSamplingRate(),
        inputParser.getMaxDepth());
    const auto wrappedRender = [renderer]() -> const std::vector<Vec3> {
        return renderer->render();
    };
    const auto image = measure(std::move(wrappedRender));

    std::ostringstream filename;
    filename << inputParser.getSceneName() << "D" << +inputParser.getMaxDepth() << "S"
        << +inputParser.getSamplingRate();
    saveImage(image, sceneData.getHeight(), sceneData.getWidth(), filename.str());

    return 0;
}
