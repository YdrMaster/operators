#ifndef __CNNL_SWIGLU_H__
#define __CNNL_SWIGLU_H__

#include "../../../operators.h"

typedef struct SwigluBangDescriptor {
    Device device;
    SwigluBangDescriptor(Device device);
} SwigluBangDescriptor;

void swiglu_cnnl_f16(Tensor gate, Tensor up, void *stream);

#endif// __CNNL_SWIGLU_H__
