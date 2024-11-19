#include "../../../devices/cuda/common_cuda.h"
#include "pooling.cuh"

<<<<<<< HEAD
infiniopStatus_t pooling_nv_gpu(PoolingCudaDescriptor_t desc, void *y, void const *x, void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id, (cudaStream_t) stream,
=======
infiniopStatus_t pooling_nv_gpu(PoolingCudaDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x) {
    checkCudaError(cudaSetDevice(desc->device_id));
    checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id,
>>>>>>> 561f952 (Add Pooling (CUDA))
                              [&](cudnnHandle_t handle) { return cudnnPoolingForward(handle, desc->pool_desc,
                                                                                     &desc->alpha, desc->x_desc, x, &desc->beta,
                                                                                     desc->y_desc, y); }));
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaPooling(PoolingCudaDescriptor_t desc,
<<<<<<< HEAD
                             void *workspace, uint64_t workspace_size,
                             void *y, void const *x, void *stream) {
    if (desc->dtype == F16 || desc->dtype == F32) {
        return pooling_nv_gpu(desc, y, x, stream);
=======
                             void *y, void const *x, void *stream) {

    if (desc->dtype == F16 || desc->dtype == F32) {
        return pooling_nv_gpu(desc, nullptr, 0, y, x);
>>>>>>> 561f952 (Add Pooling (CUDA))
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
