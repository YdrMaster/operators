#include "add.cuh"
#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"

infiniopStatus_t cudaCreateAddDescriptor(CudaHandle_t handle,
                                         AddCudaDescriptor_t *desc_ptr,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b,
                                         int device_id) {
    uint64_t ndim = c->ndim;
    if (ndim > 5 || ndim != a->ndim || ndim != b->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != c->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!dtype_eq(c->dt, F16) || c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

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

    // create and set tensor descriptors for tensors a, b, and c
    cudnnTensorDescriptor_t tensor_desc;
    checkCudnnError(cudnnCreateTensorDescriptor(&tensor_desc));
    checkCudnnError(cudnnSetTensorNdDescriptor(tensor_desc, CUDNN_DATA_HALF, ndim, shape, strides));

    // set operator descriptor
    cudnnOpTensorDescriptor_t op_desc;
    checkCudnnError(cudnnCreateOpTensorDescriptor(&op_desc));
    checkCudnnError(cudnnSetOpTensorDescriptor(
        op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    *desc_ptr = new AddCudaDescriptor{
        DevNvGpu,
        c->dt,
        device_id,
        &handle->cudnn_handle,
        tensor_desc,
        op_desc,
        alpha,
        beta};

    delete[] shape;
    delete[] strides;

    return STATUS_SUCCESS;
}

infiniopStatus_t cudaDestroyAddDescriptor(AddCudaDescriptor_t desc) {
    checkCudnnError(cudnnDestroyOpTensorDescriptor(desc->op_desc));
    checkCudnnError(cudnnDestroyTensorDescriptor(desc->tensor_desc));
    desc->handle = nullptr;
    delete desc;
    return STATUS_SUCCESS;
}
