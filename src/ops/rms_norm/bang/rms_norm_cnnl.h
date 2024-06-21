#ifndef __CNNL_RMS_NORM_H__
#define __CNNL_RMS_NORM_H__

#include "../../../operators.h"

typedef struct RMSNormBangDescriptor {
    Device device;
    RMSNormBangDescriptor(Device device);
} RMSNormBangDescriptor;

void rms_norm_cnnl_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream);

#endif// __CNNL_RMS_NORM_H__
