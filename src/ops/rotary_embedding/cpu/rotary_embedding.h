#ifndef __CPU_ROTARY_EMBEDDING_H__
#define __CPU_ROTARY_EMBEDDING_H__

#include "../../../operators.h"

void rotary_embedding_cpu_f16(struct Kernel const *kn, MutTensor t, ConstTensor pos, float theta);

#endif// __CPU_RMS_NORM_H__
