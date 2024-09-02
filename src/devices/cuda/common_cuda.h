#ifndef __COMMON_CUDA_H__
#define __COMMON_CUDA_H__

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_WARP_PER_BLOCK 32
#define WARP_SIZE 32

#include <stdexcept>
#include <string>

#define checkCudaErrorWithCode(call, errorCode)          \
    do {                                                 \
        if (auto status = call; status != cudaSuccess) { \
            return errorCode;                            \
        }                                                \
    } while (0)

#define checkCudaError(call) checkCudaErrorWithCode(call, STATUS_BAD_DEVICE)

#define checkCudnnError(call)                                     \
    do {                                                          \
        if (auto status = call; status != CUDNN_STATUS_SUCCESS) { \
            return STATUS_EXECUTION_FAILED;                       \
        }                                                         \
    } while (0)

#endif// __COMMON_CUDA_H__
