#ifndef __CUDA_HANDLE_POOL_H__
#define __CUDA_HANDLE_POOL_H__

#include <cublas_v2.h>
#include "../pool.h"

const Pool<cublasHandle_t> &get_cublas_pool(); 

template<typename T>
void use_cublas(cudaStream_t stream, T const &f) {
    auto &pool = get_cublas_pool();
    auto handle = pool.pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t) stream);
    f(*handle);
    pool.push(std::move(*handle));
}

#endif // __CUDA_HANDLE_POOL_H__
