#ifndef __CPU_SWIGLU_H__
#define __CPU_SWIGLU_H__

#include "../../../operators.h"

void swiglu_cpu_f16(struct Kernel const *kn, MutTensor gate, ConstTensor up);

#endif// __CPU_SWIGLU_H__
