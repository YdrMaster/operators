#ifndef __ASCEND_NPU_RMS_NORM_H__
#define __ASCEND_NPU_RMS_NORM_H__

#include "aclnnop/aclnn_rms_norm.h"
#include "../../../operators.h"
#include "../../../devices/ascend/common.h"

void rms_norm_ascend_npu_fp16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream);

#endif
