#include "containers/Ray.hpp"

namespace tracer::containers
{

Ray::Ray(Vec3 origin, Vec3 direction)
    : origin_(origin)
    , direction_(direction)
{}

std::ostream& operator<<(std::ostream& os, const Ray& ray)
{
    return os << "origin: " << ray.origin_ << ", destination: " << ray.direction_;
}

}  // namespace tracer::containers
