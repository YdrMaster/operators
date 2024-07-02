#ifndef __ASCEND_NPU_RMS_NORM_H__
#define __ASCEND_NPU_RMS_NORM_H__

#include "aclnnop/aclnn_rms_norm.h"
#include "../../../operators.h"
#include "../../../devices/ascend/common_ascend.h"

void rms_norm_ascend_npu_fp16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif
