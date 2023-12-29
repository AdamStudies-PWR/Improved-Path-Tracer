#include "containers/Vec3.hpp"

#include <math.h>

namespace tracer::containers
{

Vec3::Vec3(double xx, double yy, double zz)
    : xx_(xx)
    , yy_(yy)
    , zz_(zz)
{}

// Iloczyn skalarny
double Vec3::dot(const Vec3& vec2) const
{
    return (xx_ * vec2.xx_) + (yy_ * vec2.yy_) + (zz_ * vec2.zz_);
}

// Normalizacja wektora
Vec3& Vec3::norm()
{
    return *this = *this * (1/sqrt(xx_*xx_ + yy_*yy_ + zz_*zz_));
}

Vec3 Vec3::mult(const Vec3& vec2) const
{
    return Vec3(xx_*vec2.xx_, yy_*vec2.yy_, zz_*vec2.zz_);
}

Vec3 Vec3::operator+ (const Vec3& vec2) const
{
    return Vec3(xx_ + vec2.xx_, yy_ + vec2.yy_, zz_ + vec2.zz_);
}

Vec3 Vec3::operator- (const Vec3& vec2) const
{
    return Vec3(xx_ - vec2.xx_, yy_ - vec2.yy_, zz_ - vec2.zz_);
}

Vec3 Vec3::operator* (double number) const
{
    return Vec3(xx_ * number, yy_ * number, zz_ * number);
}

// Iloczyn wektorowy
Vec3 Vec3::operator% (Vec3& vec2)
{
    return Vec3(yy_*vec2.zz_ - zz_*vec2.yy_, zz_*vec2.xx_ - xx_*vec2.zz_, xx_*vec2.yy_ - yy_*vec2.xx_);
}

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    return os << "x: " << vec.xx_ << ", y: " << vec.yy_ << ", z: " << vec.zz_;
}

} // namespace tracer::containers
