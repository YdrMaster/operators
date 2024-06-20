#ifndef __CNNL_ROTARY_EMBEDDING_H__
#define __CNNL_ROTARY_EMBEDDING_H__

#include "../../../operators.h"

void rotary_embedding_cnnl_f16(MutTensor t, ConstTensor pos, float theta, void *stream);

#endif// __CNNL_ROTARY_EMBEDDING_H__
