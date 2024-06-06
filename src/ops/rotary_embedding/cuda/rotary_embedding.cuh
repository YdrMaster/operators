#ifndef __NV_GPU_ROTARY_EMBEDDING_H__
#define __NV_GPU_ROTARY_EMBEDDING_H__

#include "../../../operators.h"

void rotary_embedding_nv_gpu_f16(struct Kernel const *kn, MutTensor t, ConstTensor pos, float theta);

#endif// __NV_GPU_ROTARY_EMBEDDING_H__
