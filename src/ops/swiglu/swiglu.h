#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../export.h"
#include "../../operators.h"

typedef struct SwigluDescriptor SwigluDescriptor;

__C __export void *createSwigluDescriptor(Device, void *config);
__C __export void destroySwigluDescriptor(void *descriptor);
__C __export void swiglu(void *descriptor, MutTensor gate, ConstTensor up, void *stream);

#endif
