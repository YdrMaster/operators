#include "cuda_handle.h"

infiniopStatus_t createCudaHandle(CudaHandle_t *handle_ptr, int device_id) {
    // Check if device_id is valid
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    // create cudnn handle
    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle));

    // Create a new cublas handle pool
    auto pool = Pool<cublasHandle_t>();
    cudaSetDevice(device_id);
    cublasHandle_t handle;
    cublasCreate(&handle);
    pool.push(std::move(handle));

    *handle_ptr = new CudaContext{DevNvGpu, std::move(cudnn_handle), device_id, std::move(pool)};

    return STATUS_SUCCESS;
}
