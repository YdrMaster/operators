#ifndef __NV_GPU_RMS_NORM_H__
#define __NV_GPU_RMS_NORM_H__

#include "../operators.h"

void rms_norm_nv_gpu_f16(Kernel const *, MutTensor y, ConstTensor x, ConstTensor w, float epsilon);

#endif// __NV_GPU_RMS_NORM_H__
