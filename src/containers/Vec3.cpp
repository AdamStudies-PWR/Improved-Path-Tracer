#include "containers/Vec3.hpp"

#include <math.h>

namespace tracer::containers
{

Vec3::Vec3(double xx, double yy, double zz)
    : xx_(xx)
    , yy_(yy)
    , zz_(zz)
{}

double Vec3::dot(const Vec3& input) const
{
    return xx_*input.xx_ + yy_*input.yy_ + zz_*input.zz_;
}

Vec3& Vec3::norm()
{
    return *this = *this * (1/sqrt(xx_*xx_ + yy_*yy_ + zz_*zz_));
}

Vec3 Vec3::mult(const Vec3& input) const
{
    return Vec3(xx_*input.xx_, yy_*input.yy_, zz_*input.zz_);
}

Vec3 Vec3::operator+ (const Vec3& summand) const
{
    return Vec3(xx_ + summand.xx_, yy_ + summand.yy_, zz_ + summand.zz_);
}

Vec3 Vec3::operator- (const Vec3& subtrahend) const
{
    return Vec3(xx_ - subtrahend.xx_, yy_ - subtrahend.yy_, zz_ - subtrahend.zz_);
}

Vec3 Vec3::operator* (double number) const
{
    return Vec3(xx_ * number, yy_ * number, zz_ * number);
}

Vec3 Vec3::operator% (Vec3& input)
{
    return Vec3(yy_*input.zz_ - zz_*input.yy_, zz_*input.xx_ - xx_*input.zz_, xx_*input.yy_ - yy_*input.xx_);
}

std::ostream& operator<<(std::ostream& os, const Vec3& vec)
{
    return os << "x: " << vec.xx_ << ", y: " << vec.yy_ << ", z: " << vec.zz_;
}

} // namespace tracer::containers
