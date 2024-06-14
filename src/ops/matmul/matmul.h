#ifndef MATMUL_H
#define MATMUL_H

#include "../../operators.h"

#ifdef __cplusplus
extern "C" {
#endif

void *createMatmulDescriptor(Device, void *config);

void destroyMatmulDescriptor(void *descriptor);

void matmul(void *descriptor, MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream);

#ifdef __cplusplus
}
#endif

typedef struct MatmulDescriptor {
    Device device;
} MatmulDescriptor;

#endif
