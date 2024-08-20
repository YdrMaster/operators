#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SwigluDescriptor SwigluDescriptor;
typedef SwigluDescriptor* infiniopSwiGLUDescriptor_t;

// @deprecated
__C __export void *createSwigluDescriptor(Device, void *config);
// @deprecated
__C __export void destroySwigluDescriptor(SwigluDescriptor *descriptor);
// @deprecated
__C __export void swiglu(SwigluDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif
