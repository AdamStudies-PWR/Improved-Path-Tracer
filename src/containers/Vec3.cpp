#include "containers/Vec3.hpp"


namespace tracer::containers
{

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    return os << "x: " << vec.xx_ << ", y: " << vec.yy_ << ", z: " << vec.zz_;
}

} // namespace tracer::containers
