#ifndef MATMUL_H
#define MATMUL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MatmulDescriptor MatmulDescriptor;

__C __export MatmulDescriptor *createMatmulDescriptor(Device, void *config);

__C __export void destroyMatmulDescriptor(MatmulDescriptor *descriptor);

__C __export void matmul(MatmulDescriptor *descriptor, MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream);

#endif
