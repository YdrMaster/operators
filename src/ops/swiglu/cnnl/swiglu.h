#ifndef __CAMBRICON_MLU_SWIGLU_H__
#define __CAMBRICON_MLU_SWIGLU_H__

#include "../../../operators.h"

void swiglu_cambricon_mlu_f16(MutTensor gate, ConstTensor up, void *stream);

#endif// __CAMBRICON_MLU_SWIGLU_H__
