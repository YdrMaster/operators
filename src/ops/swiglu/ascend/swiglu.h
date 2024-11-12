#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "operators.h"
#include "../../utils.h"
#include <acl/acl_base.h>
#include <acl/acl.h>


struct SwiGLUAscendDescriptor {
    Device device;
    int device_id;
    aclDataType dtype;
    int32_t seq_len;
    int32_t di;
    int32_t sta;
    int32_t stb;
    int32_t stc;
};

typedef struct SwiGLUAscendDescriptor *SwiGLUAscendDescriptor_t;

infiniopStatus_t ascendCreateSwiGLUDescriptor(AscendHandle_t handle,
                                              SwiGLUAscendDescriptor_t *desc_ptr,
                                              infiniopTensorDescriptor_t c_desc,
                                              infiniopTensorDescriptor_t a_desc,
                                              infiniopTensorDescriptor_t b_desc);

infiniopStatus_t ascendSwiGLU(SwiGLUAscendDescriptor_t desc,
                              void *c,
                              void const *a,
                              void const *b,
                              void *stream);

infiniopStatus_t ascendDestroySwiGLUDescriptor(SwiGLUAscendDescriptor_t desc);

extern "C" void swiglu_kernel_do(void *c, void *a, void *b,
                                 float beta, int32_t nt, int32_t dh,
                                 int32_t sta, int32_t stb, int32_t stc,
                                 int dtype, void *stream);

#endif
