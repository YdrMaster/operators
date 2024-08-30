#include "add.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateAddDescriptor(infiniopHandle_t handle,
                                         AddCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b) {
    uint64_t ndim = c->ndim;
    if (ndim > 5 || ndim != a->ndim || ndim != b->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != c->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!dtype_eq(c->dt, F16) || !dtype_eq(a->dt, F16) || !dtype_eq(b->dt, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    // create cudnn handle
    cudnnHandle_t handle_ptr;
    checkCudnnError(cudnnCreate(&handle_ptr));

    // promote to dimension 4 if dimension is less than 4
    ndim = std::max(4UL, ndim);
    const auto &old_dim = a->ndim;

    // convert shape and stride arrays to int32_t
    int32_t *shape = new int32_t[ndim];
    int32_t *strides = new int32_t[ndim];
    for (size_t i = 0; i < ndim; ++i) {
        shape[i] = i < old_dim ? static_cast<int32_t>(c->shape[i]) : 1;
        strides[i] = i < old_dim ? static_cast<int32_t>(c->strides[i]) : 1;
    }

    *desc_ptr = new AddCudaDescriptor{
        DevNvGpu,
        c->dt,
        handle_ptr,
        ndim,
        shape,
        strides};

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyAddDescriptor(AddCudaDescriptor_t desc) {
    cudnnDestroy(desc->handle);
    delete desc->shape;
    delete desc->strides;
    delete desc;
    return STATUS_SUCCESS;
}
