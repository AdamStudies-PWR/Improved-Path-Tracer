#include <functional>
#include <iostream>
#include <memory>

#include "containers/Vec.hpp"
#include "renderer/Renderer.hpp"
#include "scene/SceneData.hpp"
#include "utils/Image.hpp"
#include "utils/Measurements.hpp"

using namespace tracer;

int main(int argc, char *argv[])
{
    data::SceneData sceneData;
    sceneData.initScene();

    auto renderer = std::make_shared<renderer::Renderer>(sceneData, 720, 1280, 10); //760, 1024, 1250);
    auto wrappedRender = [renderer]() -> containers::Vec* {
        return renderer->render();
    };

    auto* image = utils::measure(wrappedRender);
    utils::saveImage(image, 720, 1280);

    return 0;
}
