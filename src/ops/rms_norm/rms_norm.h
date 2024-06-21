#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RMSNormDescriptor RMSNormDescriptor;

__C __export void *
createRMSNormDescriptor(Device, void *config);
__C __export void destroyRMSNormDescriptor(void *descriptor);
__C __export void rmsNorm(void *descriptor, MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream);

#endif
