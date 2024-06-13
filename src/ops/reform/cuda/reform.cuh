#ifndef __NV_GPU_REFORM_H__
#define __NV_GPU_REFORM_H__

#include "../../../operators.h"

typedef struct ReformCudaDescriptor {
    Device device;
} ReformCudaDescriptor;

void reform_nv_gpu(MutTensor y, ConstTensor x, void *stream);

#endif// __NV_GPU_REFORM_H__
