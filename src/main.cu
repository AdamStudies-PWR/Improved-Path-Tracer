#include <functional>
#include <iostream>
#include <memory>

#include "containers/Vec3.hpp"
#include "renderer/RenderController.hpp"
#include "scene/SceneData.hpp"
#include "utils/CudaUtils.hpp"
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
    if (not checkCudaSupport())
    {
        return 0;
    }

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

    std::ostringstream filename;
    filename << inputParser.getSceneName() << "D" << +inputParser.getMaxDepth() << "S"
        << +inputParser.getSamplingRate();

    auto controller = std::make_shared<RenderController>(sceneData, inputParser.getSamplingRate(),
        inputParser.getMaxDepth());
    const auto wrappedRender = [controller, sceneData]() -> const std::vector<Vec3> {
        return controller->start(sceneData.getObjectsData());
    };
    const auto image = measure(filename.str(), std::move(wrappedRender));

    cudaError_t maybeError = cudaGetLastError();
    if (maybeError != cudaSuccess)
    {
        return 1;
    }

    saveImage(image, sceneData.getHeight(), sceneData.getWidth(), filename.str());

    return 0;
}
