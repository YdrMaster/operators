#ifndef __CAMBRICON_MLU_RMS_NORM_H__
#define __CAMBRICON_MLU_RMS_NORM_H__

#include "../../../operators.h"

void rms_norm_cambricon_mlu_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void* stream);

#endif// __CAMBRICON_MLU_RMS_NORM_H__
