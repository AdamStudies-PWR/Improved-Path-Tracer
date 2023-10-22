#include <chrono>
#include <functional>
#include <iostream>
#include <memory>

#include "utils/Vec.hpp"
#include "renderer/Renderer.hpp"
#include "scene/Scene.hpp"

using namespace tracer;

namespace
{
inline double clamp(double x)
{
    return x < 0 ? 0 : x > 1 ? 1 : x;
}

inline int toInt(double x)
{
    return int(pow(clamp(x), 1/2.2) * 255 + 0.5);
}
}  // namespace

utils::Vec* measure(std::function<utils::Vec*()> testable)
{
    std::cout << __func__ << " - Begining render..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto* image = testable();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << __func__ << " - Done" << std::endl;
    std::cout << __func__ << " - Render took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds" << std::endl;

    return image;
}

// Refactor this
void saveImage(utils::Vec* image, int height, int width)
{
    std::cout << __func__ << " - saving render..." << std::endl;

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i=0; i<(width*height); i++)
        fprintf(f,"%d %d %d ", toInt(image[i].xx_), toInt(image[i].yy_), toInt(image[i].zz_));

    std::cout << __func__ << " - Done" << std::endl;
}

int main(int argc, char *argv[])
{
    scene::Scene sceneData;
    sceneData.initScene();

    auto renderer = std::make_shared<renderer::Renderer>(sceneData, 720, 1280, 10); //760, 1024, 1250);
    auto wrappedRender = [renderer]() -> utils::Vec* {
        return renderer->render();
    };

    auto* image = measure(wrappedRender);
    saveImage(image, 720, 1280);

    return 0;
}
