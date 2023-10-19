#include "utils/Vec.hpp"

#include <math.h>

namespace tracer::utils
{

Vec::Vec(double xx, double yy, double zz)
    : xx_(xx)
    , yy_(yy)
    , zz_(zz)
{}

double Vec::dot(const Vec& input) const
{
    return xx_*input.xx_ + yy_*input.yy_ + zz_*input.zz_;
}

Vec& Vec::norm()
{
    return *this = *this * (1/sqrt(xx_*xx_ + yy_*yy_ + zz_*zz_));
}

Vec Vec::mult(const Vec& input) const
{
    return Vec(xx_*input.xx_, yy_*input.yy_, zz_*input.zz_);
}

Vec Vec::operator+ (const Vec& summand) const
{
    return Vec(xx_ + summand.xx_, yy_ + summand.yy_, zz_ + summand.zz_);
}

Vec Vec::operator- (const Vec& subtrahend) const
{
    return Vec(xx_ - subtrahend.xx_, yy_ - subtrahend.yy_, zz_ - subtrahend.zz_);
}

Vec Vec::operator* (double number) const
{
    return Vec(xx_ * number, yy_ * number, zz_ * number);
}

Vec Vec::operator% (Vec& input)
{
    return Vec(yy_*input.zz_ - zz_*input.yy_, zz_*input.xx_ - xx_*input.zz_, xx_*input.yy_ - yy_*input.xx_);
}

} // namespace tracer::utils
