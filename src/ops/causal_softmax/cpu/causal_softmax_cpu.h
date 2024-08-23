#ifndef __CPU_CAUSAL_SOFTMAX_H__
#define __CPU_CAUSAL_SOFTMAX_H__

#include "operators.h"
struct CausalSoftmaxCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t batch_size;
    uint64_t stride_b;
    uint64_t seq_len;
    uint64_t stride_i;
    uint64_t total_seq_len;
    uint64_t stride_j;
};

typedef struct CausalSoftmaxCpuDescriptor *CausalSoftmaxCpuDescriptor_t;

infiniopStatus_t cpuCreateCausalSoftmaxDescriptor(infiniopHandle_t,
                                                  CausalSoftmaxCpuDescriptor_t *,
                                                  infiniopTensorDescriptor_t y_desc);

infiniopStatus_t cpuGetCausalSoftmaxWorkspaceSize(CausalSoftmaxCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuCausalSoftmax(CausalSoftmaxCpuDescriptor_t desc,
                                  void *workspace,
                                  uint64_t workspace_size,
                                  void *data, 
                                  void *stream);

infiniopStatus_t cpuDestroyCausalSoftmaxDescriptor(CausalSoftmaxCpuDescriptor_t desc);

#endif
