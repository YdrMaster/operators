#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SwiGLUDescriptor {
    Device device;
} SwiGLUDescriptor;

typedef SwiGLUDescriptor *infiniopSwiGLUDescriptor_t;

__C __export infiniopStatus_t infiniopCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                                             infiniopSwiGLUDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t c_desc,
                                                             infiniopTensorDescriptor_t a_desc,
                                                             infiniopTensorDescriptor_t b_desc);

__C __export infiniopStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc,
                                             void *c,
                                             void *a,
                                             void *b,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc);

// // @deprecated
// __C __export void *createSwigluDescriptor(Device, void *config);
// // @deprecated
// __C __export void destroySwigluDescriptor(SwigluDescriptor *descriptor);
// // @deprecated
// __C __export void swiglu(SwigluDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif
