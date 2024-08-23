#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RMSNormDescriptor RMSNormDescriptor;
typedef RMSNormDescriptor* infiniopRMSNormDescriptor_t;

// @deprecated
__C __export void *createRMSNormDescriptor(Device, void *config);
// @deprecated
__C __export void destroyRMSNormDescriptor(RMSNormDescriptor *descriptor);
// @deprecated
__C __export void rmsNorm(RMSNormDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif
