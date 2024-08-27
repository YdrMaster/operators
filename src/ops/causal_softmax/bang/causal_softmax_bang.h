#ifndef __BANG_CAUSAL_SOFTMAX_H__
#define __BANG_CAUSAL_SOFTMAX_H__

#include "../../utils.h"
#include "operators.h"

struct CausalSoftmaxBangDescriptor {
    Device device;
    DT dtype;
    int ndim;
    int* stride;
    int* shape;
    int n;
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
