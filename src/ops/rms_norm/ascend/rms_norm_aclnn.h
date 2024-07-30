#ifndef __ASCEND_NPU_RMS_NORM_H__
#define __ASCEND_NPU_RMS_NORM_H__

#include "../../../devices/ascend/tensor_desc.h"
#include "../../../operators.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm.h"


struct RMSNormAclnnDescriptor {
    Device device;
    aclnnTensorDesc_t xDesc, wDesc, yDesc;

    RMSNormAclnnDescriptor(Device device);
    void createAclnnDescriptors() {
        aclnnCreateTensorDescriptor(&xDesc);
        aclnnCreateTensorDescriptor(&wDesc);
        aclnnCreateTensorDescriptor(&yDesc);
    }
    void destoryAclnnDescriptors() {
        aclnnDestoryTensorDescriptor(xDesc);
        aclnnDestoryTensorDescriptor(wDesc);
        aclnnDestoryTensorDescriptor(yDesc);
    }
};

void rms_norm_aclnn_f16(RMSNormAclnnDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream)

#endif
