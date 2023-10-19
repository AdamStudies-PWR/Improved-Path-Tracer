#include "utils/Ray.hpp"

namespace tracer::utils
{

Ray::Ray(Vec oo, Vec dd)
    : oo_(oo)
    , dd_(dd)
{}

}  // namespace tracer::utils
