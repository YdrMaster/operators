#ifndef CAUSAL_SOFTMAX_H
#define CAUSAL_SOFTMAX_H

#include "../../export.h"
#include "../../operators.h"

typedef struct CausalSoftmaxDescriptor CausalSoftmaxDescriptor;
typedef CausalSoftmaxDescriptor *infiniopCausalSoftmaxDescriptor_t;

__C __export infiniopStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, uint64_t *size);

__C __export infiniopStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc, void *workspace, uint64_t workspace_size, void *output_data, void *input_data, void *stream);

__C __export infiniopStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc);


// @deprecated
__C __export CausalSoftmaxDescriptor *createCausalSoftmaxDescriptor(Device, void *config);
// @deprecated
__C __export void destroyCausalSoftmaxDescriptor(CausalSoftmaxDescriptor *descriptor);
// @deprecated
__C __export void causalSoftmax(CausalSoftmaxDescriptor *descriptor, Tensor y, void *stream);


#endif
