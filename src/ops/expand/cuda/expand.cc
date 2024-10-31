#include "expand.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateExpandDescriptor(CudaHandle_t handle,
                                            ExpandCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (!isValidBroadcastShape(y, x)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for x in terms of y
    int64_t *x_strides = new int64_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] = (i < ndim - x->ndim || y->shape[i] != x->shape[i + x->ndim - ndim]) ? 0 : x->strides[i + x->ndim - ndim];
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, handle->device_id);

    int64_t *x_strides_d, *y_strides_d;
    uint64_t *y_shape_d;
    checkCudaErrorWithCode(cudaMalloc(&x_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc(&y_strides_d, ndim * sizeof(int64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMalloc(&y_shape_d, ndim * sizeof(uint64_t)), STATUS_MEMORY_NOT_ALLOCATED);
    checkCudaErrorWithCode(cudaMemcpy(x_strides_d, x_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(y_strides_d, y->strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);
    checkCudaErrorWithCode(cudaMemcpy(y_shape_d, y->shape, ndim * sizeof(uint64_t), cudaMemcpyHostToDevice), STATUS_EXECUTION_FAILED);

    *desc_ptr = new ExpandCudaDescriptor{
        DevNvGpu,
        y->dt,
        handle->device_id,
        ndim,
        y_data_size,
        static_cast<uint64_t>(prop.maxGridSize[0]),
        y_shape_d,
        x_strides_d,
        y_strides_d,
    };

    delete[] x_strides;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyExpandDescriptor(ExpandCudaDescriptor_t desc) {
    cudaFree((void *) desc->x_strides);
    cudaFree((void *) desc->y_strides);
    cudaFree((void *) desc->y_shape);
    delete desc;
    return STATUS_SUCCESS;
}
