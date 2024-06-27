#include "handle_pool.h"
#include <vector>
#include <cuda_runtime.h>

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
