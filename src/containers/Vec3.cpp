#include "containers/Vec3.hpp"


namespace tracer::containers
{

double Vec3::length() const
{
    return sqrt(pow(xx_, 2) + pow(yy_, 2) + pow(zz_, 2));
}

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    return os << "x: " << vec.xx_ << ", y: " << vec.yy_ << ", z: " << vec.zz_;
}

} // namespace tracer::containers
