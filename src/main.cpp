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

    auto renderer = std::make_shared<Renderer>(sceneData, inputParser.getSamplingRate());
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
    j["camera"]["direction"] = {{"xx", camera.dd_.xx_}, {"yy", camera.dd_.yy_}, {"zz", camera.dd_.zz_}};

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
