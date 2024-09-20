#include "cuda_handle.h"

infiniopStatus_t createCudaHandle(CudaHandle_t *handle_ptr, int device_id) {
    // Check if device_id is valid
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_id >= device_count) {
        return STATUS_BAD_DEVICE;
    }

    // Create a new cublas handle pool
    auto pool = std::make_shared<Pool<cublasHandle_t>>();
    if (cudaSetDevice(device_id) != cudaSuccess) {
        return STATUS_BAD_DEVICE;
    }
    cublasHandle_t handle;
    cublasCreate(&handle);
    pool->push(std::move(handle));

    // create a cudnn handle pool
    auto cudnn_pool = std::make_shared<Pool<cudnnHandle_t>>();
    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle));
    cudnn_pool->push(std::move(cudnn_handle));

    *handle_ptr = new CudaContext{DevNvGpu, device_id, std::move(pool), std::move(cudnn_pool)};

    return STATUS_SUCCESS;
}

infiniopStatus_t deleteCudaHandle(CudaHandle_t handle_ptr) {
    handle_ptr->cublas_handles_t = nullptr;
    handle_ptr->cudnn_handles_t = nullptr;
    delete handle_ptr;

    return STATUS_SUCCESS;
}
