#ifndef MATMUL_H
#define MATMUL_H

#include "../../export.h"
#include "../../operators.h"

typedef struct MatmulDescriptor MatmulDescriptor;
typedef MatmulDescriptor* infiniopMatmulDescriptor_t;

// @deprecated
__C __export MatmulDescriptor *createMatmulDescriptor(Device, void *config);
// @deprecated
__C __export void destroyMatmulDescriptor(MatmulDescriptor *descriptor);
// @deprecated
__C __export void matmul(MatmulDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream);

#endif