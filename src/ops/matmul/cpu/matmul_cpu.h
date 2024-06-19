#ifndef __CPU_MATMUL_H__
#define __CPU_MATMUL_H__

#include "../../../operators.h"
typedef struct MatmulCpuDescriptor {
    Device device;
} MatmulCpuDescriptor;

void matmul_cpu_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha);

#endif// __CPU_MATMUL_H__
