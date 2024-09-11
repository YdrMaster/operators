#ifndef __CUDA_ADD_H__
#define __CUDA_ADD_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cudnn.h>

struct AddCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    cudnnHandle_t *handle;
    cudnnTensorDescriptor_t tensor_desc;
    cudnnOpTensorDescriptor_t op_desc;
    const float alpha;
    const float beta;
};

typedef struct AddCudaDescriptor *AddCudaDescriptor_t;

infiniopStatus_t cudaCreateAddDescriptor(CudaHandle_t,
                                         AddCudaDescriptor_t *,
                                         infiniopTensorDescriptor_t c,
                                         infiniopTensorDescriptor_t a,
                                         infiniopTensorDescriptor_t b);

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream);

infiniopStatus_t cudaDestroyAddDescriptor(AddCudaDescriptor_t desc);

#endif
