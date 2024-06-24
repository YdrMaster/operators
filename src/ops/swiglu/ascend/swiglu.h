#ifndef __ASCEND_NPU_SWIGLU_H__
#define __ASCEND_NPU_SWIGLU_H__

#include "../../../operators.h"

void swiglu_ascend_npu_fp16(MutTensor gate, ConstTensor up, void *stream);

#endif