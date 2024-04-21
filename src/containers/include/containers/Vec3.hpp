#pragma once

#include <math.h>
#include <ostream>
#include <stdint.h>

namespace tracer::containers
{

#ifndef __device__
#define __device__
#endif

#ifndef __host__
#define __host__
#endif

struct Vec3
{
    __host__ __device__ Vec3(double xx=0.0, double yy=0.0, double zz=0.0)
        : xx_(xx)
        , yy_(yy)
        , zz_(zz)
    {}

    // Iloczyn skalarny
    __device__ double dot(const Vec3& vec2) const
    {
        return (xx_ * vec2.xx_) + (yy_ * vec2.yy_) + (zz_ * vec2.zz_);
    }

    __device__ Vec3 mult(const Vec3& vec2) const
    {
        return Vec3(xx_*vec2.xx_, yy_*vec2.yy_, zz_*vec2.zz_);
    }

    __host__ __device__ double distance(const Vec3& vec2) const
    {
        return sqrt(pow((xx_ - vec2.xx_), 2) + pow((yy_ - vec2.yy_), 2) + pow((zz_ - vec2.zz_), 2));
    }

    __device__ double length() const
    {
        return sqrt(pow(xx_, 2) + pow(yy_, 2) + pow(zz_, 2));
    }

    // Normalizacja wektora
    __host__ __device__ Vec3& norm()
    {
        return *this = *this * (1/sqrt(xx_*xx_ + yy_*yy_ + zz_*zz_));
    }

    __host__ __device__ Vec3 operator+ (const Vec3& vec2) const
    {
        return Vec3(xx_ + vec2.xx_, yy_ + vec2.yy_, zz_ + vec2.zz_);
    }

    __device__ Vec3 operator- (const Vec3& vec2) const
    {
        return Vec3(xx_ - vec2.xx_, yy_ - vec2.yy_, zz_ - vec2.zz_);
    }

    __host__ __device__ Vec3 operator* (double number) const
    {
        return Vec3(xx_ * number, yy_ * number, zz_ * number);
    }

    // Iloczyn wektorowy
    __host__ __device__ Vec3 operator% (Vec3& vec2)
    {
        return Vec3(yy_*vec2.zz_ - zz_*vec2.yy_, zz_*vec2.xx_ - xx_*vec2.zz_, xx_*vec2.yy_ - yy_*vec2.xx_);
    }

    __host__ __device__ bool operator== (const Vec3& vec2) const
    {
        return xx_ == vec2.xx_ and yy_ == vec2.yy_ and zz_ == vec2.zz_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3& vec);

    double xx_;
    double yy_;
    double zz_;
};

}  // namespace tracer::containers
