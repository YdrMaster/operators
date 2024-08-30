#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <stdexcept>
#include <string>

#define checkCudaError(call)                                                \
    if (auto err = call; err != cudaSuccess)                                \
    throw std::runtime_error(std::string("[") + __FILE__ + ":" +           \
                              std::to_string(__LINE__) + "] CUDA error (" + \
                              #call + "): " + cudaGetErrorString(err))

#define checkCudnnError(call)                                               \
    if (auto err = call; err != CUDNN_STATUS_SUCCESS)                       \
    throw std::runtime_error(std::string("[") + __FILE__ + ":" +            \
                             std::to_string(__LINE__) + "] cuDNN error (" + \
                             #call + "): " + cudnnGetErrorString(err))

#endif// __COMMON_CUDA_H__
