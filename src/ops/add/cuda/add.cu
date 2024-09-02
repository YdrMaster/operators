#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "add.cuh"

infiniopStatus_t add_nv_gpu_f16(AddCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    checkCudnnError(cudnnOpTensor(*desc->handle, desc->op_desc, &desc->alpha,
                                  desc->tensor_desc, a, &desc->alpha, desc->tensor_desc, b,
                                  &desc->beta, desc->tensor_desc, c));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        return add_nv_gpu_f16(desc, c, a, b, stream);
    }

    return STATUS_BAD_TENSOR_DTYPE;
}
