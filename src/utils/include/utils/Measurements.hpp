#pragma once

#include <functional>

#include "containers/Vec.hpp"

namespace tracer::utils
{

containers::Vec* measure(std::function<containers::Vec*()> testable);

}  // namespace tracer::utils
