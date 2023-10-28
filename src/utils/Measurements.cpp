#include "utils/Measurements.hpp"

#include <chrono>
#include <iostream>

namespace tracer::utils
{

containers::Vec* measure(std::function<containers::Vec*()> testable)
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

}  // namespace tracer::utils
