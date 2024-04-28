#pragma once

#include <functional>
#include <string>
#include <vector>

#include "containers/Vec3.hpp"

namespace tracer::utils
{

const std::vector<containers::Vec3> measure(const std::string& filename,
    std::function<std::vector<containers::Vec3>()> testable);

}  // namespace tracer::utils
