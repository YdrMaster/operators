#ifndef __CNNL_SWIGLU_H__
#define __CNNL_SWIGLU_H__

#include "../../../operators.h"

void swiglu_cnnl_f16(MutTensor gate, ConstTensor up, void *stream);

#endif// __CNNL_SWIGLU_H__
