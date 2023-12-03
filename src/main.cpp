#include <functional>
#include <iostream>
#include <memory>

#include "containers/Vec.hpp"
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

    auto renderer = std::make_shared<Renderer>(sceneData, 720, 1280, inputParser.getSamplingRate());
    const auto wrappedRender = [renderer]() -> Vec* {
        return renderer->render();
    };

    auto* image = measure(std::move(wrappedRender));
    saveImage(image, 720, 1280);

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
    j["camera"]["position"] = {{"xx", camera.oo_.xx_}, {"yy", camera.oo_.yy_}, {"zz", camera.oo_.zz_}};
    j["camera"]["direction"] = {{"xx", camera.oo_.xx_}, {"yy", camera.oo_.yy_}, {"zz", camera.oo_.zz_}};

    std::vector<json> obj_array;
    for (const auto obj : sceneData.spheres_)
    {
        json temp = {
            {"type", "sphere"},
            {"radius", obj.radius_},
            {"position", {
                {"xx", obj.position_.xx_},
                {"yy", obj.position_.yy_},
                {"zz", obj.position_.zz_}
            }},
            {"emission", {
                {"xx", obj.emission_.xx_},
                {"yy", obj.emission_.yy_},
                {"zz", obj.emission_.zz_}
            }},
            {"color", {
                {"xx", obj.color_.xx_},
                {"yy", obj.color_.yy_},
                {"zz", obj.color_.zz_}
            }},
            {"reflection", obj.reflection_}
        };

        obj_array.push_back(temp);
    }

    j["objects"] = obj_array;
    std::ofstream o("spheres.json");
    o << std::setw(4) << j << std::endl;

*/
