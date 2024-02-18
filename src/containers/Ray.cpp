#include "containers/Ray.hpp"

namespace tracer::containers
{

std::ostream& operator<<(std::ostream& os, const Ray& ray)
{
    return os << "origin: " << ray.origin_ << ", destination: " << ray.direction_;
}

}  // namespace tracer::containers
