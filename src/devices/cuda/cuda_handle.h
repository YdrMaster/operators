#ifndef CUDA_HANDLE_H
#define CUDA_HANDLE_H

#include "../pool.h"
#include "common_cuda.h"
#include "device.h"
#include "status.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

struct CudaContext {
    Device device;
    cudnnHandle_t cudnn_handle;
    int device_id;
    Pool<cublasHandle_t> cublas_handles;
};
typedef struct CudaContext *CudaHandle_t;

infiniopStatus_t createCudaHandle(CudaHandle_t *handle_ptr, int device_id);


template<typename T>
void use_cublas(CudaHandle_t cuda_handle, cudaStream_t stream, T const &f) {
    auto &pool = cuda_handle->cublas_handles;
    auto handle = pool.pop();
    if (!handle) {
        cudaSetDevice(cuda_handle->device_id);
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t) stream);
    f(*handle);
    pool.push(std::move(*handle));
}


#endif
