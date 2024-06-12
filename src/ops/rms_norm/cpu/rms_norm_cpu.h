#ifndef __CPU_RMS_NORM_H__
#define __CPU_RMS_NORM_H__

#include "../../../operators.h"

void rms_norm_cpu_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon);

#endif// __CPU_RMS_NORM_H__
