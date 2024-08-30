#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "add.cuh"

void add_nv_gpu_f16(AddCudaDescriptor_t desc, void *c, void *a, void *b, void *stream) {
    // Create and set tensor descriptors for tensors a, b, and c
    cudnnTensorDescriptor_t tensorDesc;
    checkCudnnError(cudnnCreateTensorDescriptor(&tensorDesc));
    checkCudnnError(cudnnSetTensorNdDescriptor(tensorDesc, CUDNN_DATA_HALF, desc->ndim, desc->shape, desc->strides));

    cudnnOpTensorDescriptor_t opDesc;
    checkCudnnError(cudnnCreateOpTensorDescriptor(&opDesc));
    checkCudnnError(cudnnSetOpTensorDescriptor(
        opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

    // Perform the addition
    const float alpha = 1.0f;
    const float beta = 0.0f;
    checkCudnnError(cudnnOpTensor(desc->handle, opDesc, &alpha,
                                  tensorDesc, a, &alpha, tensorDesc, b,
                                  &beta, tensorDesc, c));

    // Clean up
    checkCudnnError(cudnnDestroyOpTensorDescriptor(opDesc));
    checkCudnnError(cudnnDestroyTensorDescriptor(tensorDesc));
}

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *workspace,
                         unsigned long int workspace_size,
                         void *c, void *a, void *b,
                         void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        add_nv_gpu_f16(desc, c, a, b, stream);
        return STATUS_SUCCESS;
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
