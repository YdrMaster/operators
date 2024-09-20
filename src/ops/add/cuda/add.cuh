#ifndef __CUDA_ADD_H__
#define __CUDA_ADD_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct AddCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t c_data_size;
    uint64_t max_grid_size;
    int64_t const *a_strides;
    int64_t const *b_strides;
    int64_t const *c_strides;
    bool broadcasted;
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
