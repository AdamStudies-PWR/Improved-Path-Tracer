#include "containers/Ray.hpp"

namespace tracer::containers
{

Ray::Ray(Vec oo, Vec dd)
    : oo_(oo)
    , dd_(dd)
{}

std::ostream& operator<<(std::ostream& os, const Ray& ray)
{
    return os << "origin: " << ray.oo_ << ", destination: " << ray.dd_;
}

}  // namespace tracer::containers
