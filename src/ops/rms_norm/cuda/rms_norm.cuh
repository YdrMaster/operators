#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "../../../operators.h"

typedef struct RMSNormCudaDescriptor {
    Device device;
} RMSNormCudaDescriptor;

void rms_norm_nv_gpu_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif// __NV_GPU_RMS_NORM_H__
