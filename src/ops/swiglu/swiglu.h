#ifndef SWIGLU_H
#define SWIGLU_H

#include "../../operators.h"

#ifdef __cplusplus
extern "C" {
#endif

void *createSwigluDescriptor(Device, void *config);

void destroySwigluDescriptor(void *descriptor);

void swiglu(void *descriptor, MutTensor gate, ConstTensor up, void *stream);

#ifdef __cplusplus
}
#endif

typedef struct SwigluDescriptor {
    Device device;
} SwigluDescriptor;

#endif
