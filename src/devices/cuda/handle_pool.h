#ifndef __CUDA_HANDLE_POOL_H__
#define __CUDA_HANDLE_POOL_H__

#include <cublas_v2.h>
#include <mutex>
#include <vector>
#include "../pool.h"

const Pool<cublasHandle_t> &get_cublas_pool() {
    int device_id;
    cudaGetDevice(&device_id);
    static std::once_flag flag;
    static std::vector<Pool<cublasHandle_t>> cublas_pool;
    std::call_once(flag, [&]() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; i++) {
            auto pool = Pool<cublasHandle_t>();
            cublasHandle_t handle;
            cublasCreate(&handle);
            pool.push(std::move(handle));
            cublas_pool.emplace_back(std::move(pool));
        }
    });
    return cublas_pool[device_id];
}

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
