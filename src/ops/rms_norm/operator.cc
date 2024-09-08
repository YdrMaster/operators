#include "../utils.h"
#include "ops/rms_norm/rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rms_norm.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rms_norm_bang.h"
#include "bang/rms_norm_cnnl.h"
#endif
#ifdef ENABLE_ASCEND_NPU
#include "ascend/rms_norm_aclnn.h"
#endif


__C infiniopStatus_t infiniopCreateRMSNormDescriptor(infiniopHandle_t handle,
                                                     infiniopRMSNormDescriptor_t *desc_ptr,
                                                     infiniopTensorDescriptor_t y,
                                                     infiniopTensorDescriptor_t x,
                                                     infiniopTensorDescriptor_t w,
                                                     float eps) {
    switch (handle->device) {
#ifdef ENABLE_CPU
// TODO
#endif
#ifdef ENABLE_NV_GPU
// TODO
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            printf("in infiniopCreateRMSNormDescriptor\n");
            return aclnnCreateRMSNormDescriptor((AscendHandle_t) handle,
                                                (RMSNormAclnnDescriptor_t *) desc_ptr,
                                                y,
                                                x,
                                                w,
                                                eps);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc,
                                                     uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
// TODO
#endif
#ifdef ENABLE_NV_GPU
// TODO
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnGetRMSNormWorkspaceSize((RMSNormAclnnDescriptor_t) desc,
                                                size);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C __export infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc,
                                              void *workspace,
                                              uint64_t workspace_size,
                                              void *y,
                                              void *x,
                                              void *w,
                                              void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
// TODO
#endif
#ifdef ENABLE_NV_GPU
// TODO
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnRMSNorm((RMSNormAclnnDescriptor_t) desc,
                                workspace,
                                workspace_size,
                                y,
                                x,
                                w,
                                stream);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
// TODO
#endif
#ifdef ENABLE_NV_GPU
// TODO
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO
#endif
#ifdef ENABLE_ASCEND_NPU
        case DevAscendNpu: {
            return aclnnDestroyRMSNormDescriptor((RMSNormAclnnDescriptor_t) desc);
        }
#endif
    }
    return STATUS_BAD_DEVICE;
}
