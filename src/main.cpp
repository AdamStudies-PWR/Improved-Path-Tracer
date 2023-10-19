#include <chrono>
#include <functional>
#include <iostream>
#include <memory>

#include "utils/Vec.hpp"
#include "renderer/Renderer.hpp"

using namespace tracer;

namespace
{
int HEIGHT = 768;
int WIDTH = 1024;

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
void saveImage(utils::Vec* image)
{
    std::cout << __func__ << " - saving render..." << std::endl;

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    for (int i=0; i<WIDTH*HEIGHT; i++)
        fprintf(f,"%d %d %d ", toInt(image[i].xx_), toInt(image[i].yy_), toInt(image[i].zz_));

    std::cout << __func__ << " - Done" << std::endl;
}

int main(int argc, char *argv[])
{
    std::shared_ptr<renderer::Renderer> renderer = std::make_shared<renderer::Renderer>();
    renderer->initScene();

    auto wrappedRender = [renderer]() -> utils::Vec* {
        return renderer->render();
    };

    auto* image = measure(wrappedRender);
    saveImage(image);

    return 0;
}
