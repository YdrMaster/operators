#ifndef __ASCEND_NPU_SWIGLU_H__
#define __ASCEND_NPU_SWIGLU_H__

#include "acl/acl.h"
#include "../../../operators.h"
#include "../../../devices/ascend/common_ascend.h"
// #include "../../utils.h"

void swiglu_ascend_npu_fp16(Tensor gate, Tensor up, void *stream);

#endif