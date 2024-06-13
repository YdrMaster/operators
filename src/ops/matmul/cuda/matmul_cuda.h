#ifndef __NV_GPU_MATMUL_H__
#define __NV_GPU_MATMUL_H__

#include "../../../operators.h"

void matmul_nv_gpu_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream);

#endif// __NV_GPU_MATMUL_H__
