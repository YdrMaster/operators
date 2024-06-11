#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../operators.h"

#ifdef __cplusplus
extern "C" {
#endif

void *createRMSNormDescriptor(Device, void *config);

void destroyRMSNormDescriptor(void *descriptor);

void rmsNorm(void *descriptor, MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream);

#ifdef __cplusplus
}
#endif

typedef struct RMSNormDescriptor {
    Device device;
} RMSNormDescriptor;

#endif
