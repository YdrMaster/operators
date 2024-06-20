#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../export.h"
#include "../../operators.h"

__C __export void *createRMSNormDescriptor(Device, void *config);
__C __export void destroyRMSNormDescriptor(void *descriptor);
__C __export void rmsNorm(void *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

typedef struct RMSNormDescriptor {
    Device device;
} RMSNormDescriptor;

#endif
