#ifndef __CPU_REFORM_H__
#define __CPU_REFORM_H__

#include "../../../operators.h"

typedef struct ReformCpuDescriptor {
    Device device;
} ReformCpuDescriptor;

void reform_cpu(Tensor y, Tensor x);

#endif// __CPU_REFORM_H__
