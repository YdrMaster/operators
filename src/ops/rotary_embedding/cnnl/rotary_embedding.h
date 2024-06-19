#ifndef __CAMBRICON_MLU_ROTARY_EMBEDDING_H__
#define __CAMBRICON_MLU_ROTARY_EMBEDDING_H__

#include "../../../operators.h"

void rotary_embedding_cambricon_mlu_f16(MutTensor t, ConstTensor pos, float theta, void *stream);

#endif// __CAMBRICON_MLU_ROTARY_EMBEDDING_H__
