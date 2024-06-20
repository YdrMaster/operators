#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../export.h"
#include "../../operators.h"

__C __export void *createSwigluDescriptor(Device, void *config);
__C __export void destroySwigluDescriptor(void *descriptor);
__C __export void swiglu(void *descriptor, Tensor gate, Tensor up, void *stream);

typedef struct SwigluDescriptor {
    Device device;
} SwigluDescriptor;

#endif
