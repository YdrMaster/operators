#ifndef __CUDA_ADD_H__
#define __CUDA_ADD_H__

#include "../../../devices/cuda/common_cuda.h"
#include "operators.h"
#include <cudnn.h>

struct AddCudaDescriptor {
    Device device;
    DT dtype;
    cudnnHandle_t handle;
    uint64_t ndim;
    int32_t *shape;
    int32_t *strides;
};

typedef struct AddCudaDescriptor *AddCudaDescriptor_t;

infiniopStatus_t cudaCreateAddDescriptor(infiniopHandle_t,
                                         AddCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b);

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *workspace,
                         uint64_t workspace_size,
                         void *c, void *a, void *b,
                         void *stream);

infiniopStatus_t cudaDestroyAddDescriptor(AddCudaDescriptor_t desc);

#endif
