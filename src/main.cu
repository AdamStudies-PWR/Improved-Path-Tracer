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

    auto controller = std::make_shared<RenderContoller>(sceneData, inputParser.getSamplingRate());
    const auto wrappedRender = [controller]() -> const std::vector<Vec3> {
        return controller->start();
    };
    const auto image = measure(std::move(wrappedRender));

    cudaError_t maybeError = cudaGetLastError();
    if (maybeError != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(maybeError) << std::endl;
        return 1;
    }

    saveImage(image, sceneData.getHeight(), sceneData.getWidth());
    return 0;
}

/*
SCENE SAVING LOGIC

#include <string>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

   json j;

    j["width"] = 1280;
    j["height"] = 720;

    auto camera = sceneData.getCamera();
    j["camera"]["position"] = {{"xx", camera.origin_.xx_}, {"yy", camera.origin_.yy_}, {"zz", camera.origin_.zz_}};
    j["camera"]["direction"] = {{"xx", camera.direction_.xx_}, {"yy", camera.direction_.yy_}, {"zz", camera.direction_.zz_}};

    std::vector<json> obj_array;
    for (const auto obj : sceneData.objects_)
    {
        json temp = {
            {"type", "sphere"},
            {"radius", obj->radius_},
            {"position", {
                {"xx", obj->getPosition().xx_},
                {"yy", obj->getPosition().yy_},
                {"zz", obj->getPosition().zz_}
            }},
            {"emission", {
                {"xx", obj->getEmission().xx_},
                {"yy", obj->getEmission().yy_},
                {"zz", obj->getEmission().zz_}
            }},
            {"color", {
                {"xx", obj->getColor().xx_},
                {"yy", obj->getColor().yy_},
                {"zz", obj->getColor().zz_}
            }},
            {"reflection", obj->getReflectionType()}
        };

        obj_array.push_back(temp);
    }

    j["objects"] = obj_array;
    std::ofstream o("spheres.json");
    o << std::setw(4) << j << std::endl;

*/
