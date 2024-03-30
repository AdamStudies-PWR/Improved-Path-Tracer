#pragma once

#include <stdint.h>
#include <functional>


namespace tracer::renderer
{

void renderCPU(const uint32_t width, std::function<void(const uint32_t)> gpuCallback);

}  // namespace tracer::renderer
