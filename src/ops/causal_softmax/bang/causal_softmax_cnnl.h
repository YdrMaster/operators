#ifndef __CNNL_CAUSAL_SOFTMAX_H__
#define __CNNL_CAUSAL_SOFTMAX_H__

#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "operators.h"
#include <vector>

struct CausalSoftmaxCnnlDescriptor {
    Device device;
    DT dtype;
    BangHandle_t handle;
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t maskDesc;
    std::vector<int> dims;
};

typedef struct CausalSoftmaxCnnlDescriptor *CausalSoftmaxCnnlDescriptor_t;

infiniopStatus_t cnnlCreateCausalSoftmaxDescriptor(infiniopHandle_t handle,
                                                   CausalSoftmaxCnnlDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cnnlGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCnnlDescriptor_t desc, unsigned long int *size);

infiniopStatus_t cnnlCausalSoftmax(CausalSoftmaxCnnlDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t cnnlDestroyCausalSoftmaxDescriptor(CausalSoftmaxCnnlDescriptor_t desc);

#endif
