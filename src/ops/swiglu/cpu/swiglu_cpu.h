#ifndef __CPU_SWIGLU_H__
#define __CPU_SWIGLU_H__

#include "operators.h"

struct SwiGLUCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t seq_len;
    uint64_t di;
    uint64_t stride_a;
    uint64_t stride_b;
    uint64_t stride_c;
};

typedef struct SwiGLUCpuDescriptor *SwiGLUCpuDescriptor_t;

infiniopStatus_t cpuCreateSwiGLUDescriptor(infiniopHandle_t handle,
                                           SwiGLUCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t c_dec,
                                           infiniopTensorDescriptor_t a_desc,
                                           infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cpuSwiGLU(SwiGLUCpuDescriptor_t desc,
                           void *c,
                           void *a,
                           void *b,
                           void *stream);

infiniopStatus_t cpuDestroySwiGLUDescriptor(SwiGLUCpuDescriptor_t desc);

#endif// __CPU_SWIGLU_H__
