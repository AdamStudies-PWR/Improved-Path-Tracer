#include "utils/CudaUtils.hpp"

#include <iostream>

namespace tracer::utils
{

bool checkCudaSupport()
{
    int cudaDevices;
    cudaGetDeviceCount(&cudaDevices);

    if (cudaDevices <= 0)
    {
        std::cout << "CUDA capable device not found! Cannot continue";
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU device: " <<  prop.name << std::endl;
    return true;
}

}  // namespace tracer::utils
