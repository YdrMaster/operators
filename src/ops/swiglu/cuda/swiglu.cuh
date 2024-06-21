#ifndef __NV_GPU_SWIGLU_H__
#define __NV_GPU_SWIGLU_H__

#include "../../../operators.h"

typedef struct SwigluCudaDescriptor {
    Device device;
} SwigluCudaDescriptor;

void swiglu_nv_gpu_f16(Tensor gate, Tensor up, void *stream);

#endif// __NV_GPU_SWIGLU_H__
