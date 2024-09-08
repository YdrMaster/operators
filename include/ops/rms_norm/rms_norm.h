#ifndef RMS_NORM_H
#define RMS_NORM_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RMSNormDescriptor {
    Device device;
} RMSNormDescriptor;

typedef RMSNormDescriptor *infiniopRMSNormDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRMSNormDescriptor(infiniopHandle_t handle,
                                                              infiniopRMSNormDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              infiniopTensorDescriptor_t w,
                                                              float eps);

__C __export infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc,
                                                              uint64_t *size);

__C __export infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc,
                                              void *workspace,
                                              uint64_t workspace_size,
                                              void *y,
                                              void *x,
                                              void *w,
                                              void *stream);

__C __export infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc);

// @deprecated
__C __export void *
createRMSNormDescriptor(Device, void *config);
// @deprecated
__C __export void destroyRMSNormDescriptor(RMSNormDescriptor *descriptor);
// @deprecated
__C __export void rmsNorm(RMSNormDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream);

#endif
