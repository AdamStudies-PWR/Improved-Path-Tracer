#pragma once

#include <cstdint>

#include "containers/Vec3.hpp"


namespace tracer::scene::objects::helpers
{

#ifndef __device__
#define __device__
#endif

namespace
{
__device__ void swap(double* distances, containers::Vec3* array, const uint8_t e1, const uint8_t e2)
{
    const auto tempDistance = distances[e1];
    distances[e1] = distances[e2];
    distances[e2] = tempDistance;

    const auto tempVec3 = array[e1];
    array[e1] = array[e2];
    array[e2] = tempVec3;
}

__device__ uint8_t partition(double* distances, containers::Vec3* array, uint8_t low, const uint8_t high)
{
    double value = distances[high];
    uint8_t i = low - 1;

    for (uint8_t j = low; j <= (high - 1); j++)
    {
        if (distances[j] <= value)
        {
            i++;
            swap(distances, array, i, j);
        }
    }

    swap(distances, array, (i + 1), high);
    return i + 1;
}
}  // namespace

__device__ void quickSort(containers::Vec3* array, const containers::Vec3& refPoint, uint8_t elements)
{
    double* distances = new double[elements]();
    for (uint8_t i = 0; i < elements; i++)
    {
        distances[i] = refPoint.distance(array[i]);
    }

    uint8_t index = 0;
    uint8_t* buffer = new uint8_t[elements]();
    buffer[index++] = 0;
    buffer[index] = elements - 1;

    while (index > 0)
    {
        uint8_t high = buffer[index--];
        uint8_t low = buffer[index];

        uint8_t div = partition(distances, array, low, high);
        if (div - 1 > low)
        {
            buffer[index++] = low;
            buffer[index] = div - 1;
        }

        if (div + 1 < high)
        {
            buffer[++index] = div + 1;
            buffer[++index] = high;
        }
    }

    delete distances;
    delete buffer;
}

}  // namespace tracer::scene::objects::helpers
