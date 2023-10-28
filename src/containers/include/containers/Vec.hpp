#pragma once

namespace tracer::containers
{

class Vec
{
public:
    Vec(double xx=0, double yy=0, double zz=0);

    double dot(const Vec& input) const;
    Vec& norm();
    Vec mult(const Vec& input) const;

    Vec operator+ (const Vec& summand) const;
    Vec operator- (const Vec& subtrahend) const;
    Vec operator* (double number) const;
    Vec operator% (Vec& input);

    double xx_;
    double yy_;
    double zz_;
};

} // namespace tracer::containers
