#ifndef __NV_GPU_SWIGLU_H__
#define __NV_GPU_SWIGLU_H__

#include "../../../operators.h"

void swiglu_nv_gpu_f16(struct Kernel const *kn, MutTensor gate, ConstTensor up);

#endif// __NV_GPU_SWIGLU_H__
