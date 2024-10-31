#ifndef __CUDA_EXPAND_H__
#define __CUDA_EXPAND_H__

#include "../../../devices/cuda/common_cuda.h"
#include "../../../devices/cuda/cuda_handle.h"
#include "operators.h"
#include <cuda_fp16.h>
#include <numeric>

struct ExpandCudaDescriptor {
    Device device;
    DT dtype;
    int device_id;
    uint64_t ndim;
    uint64_t y_data_size;
    uint64_t max_grid_size;
    uint64_t const *y_shape;
    int64_t const *x_strides;
    int64_t const *y_strides;
};

typedef struct ExpandCudaDescriptor *ExpandCudaDescriptor_t;

infiniopStatus_t cudaCreateExpandDescriptor(CudaHandle_t,
                                            ExpandCudaDescriptor_t *,
                                            infiniopTensorDescriptor_t y,
                                            infiniopTensorDescriptor_t x);

infiniopStatus_t cudaExpand(ExpandCudaDescriptor_t desc,
                            void *y, void const *x,
                            void *stream);

infiniopStatus_t cudaDestroyExpandDescriptor(ExpandCudaDescriptor_t desc);

#endif
