#ifndef __BANG_CAUSAL_SOFTMAX_H__
#define __BANG_CAUSAL_SOFTMAX_H__

#include "../../../devices/bang/bang_handle.h"
#include "cnnl.h"
#include "operators.h"
#include <vector>

struct CausalSoftmaxBangDescriptor {
    Device device;
    DT dtype;
    BangHandle_t handle;
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t maskDesc;
    std::vector<int> dims;
};

typedef struct CausalSoftmaxBangDescriptor *CausalSoftmaxBangDescriptor_t;

infiniopStatus_t bangCreateCausalSoftmaxDescriptor(infiniopHandle_t handle,
                                                   CausalSoftmaxBangDescriptor_t *desc_ptr,
                                                   infiniopTensorDescriptor_t y_desc);

infiniopStatus_t bangGetCausalSoftmaxWorkspaceSize(CausalSoftmaxBangDescriptor_t desc, unsigned long int *size);

infiniopStatus_t bangCausalSoftmax(CausalSoftmaxBangDescriptor_t desc,
                                   void *workspace,
                                   unsigned long int workspace_size,
                                   void *data,
                                   void *stream);

infiniopStatus_t bangDestroyCausalSoftmaxDescriptor(CausalSoftmaxBangDescriptor_t desc);

#endif
