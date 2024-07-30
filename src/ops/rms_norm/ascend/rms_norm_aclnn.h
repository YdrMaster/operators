#ifndef __ASCEND_NPU_RMS_NORM_H__
#define __ASCEND_NPU_RMS_NORM_H__

#include "../../../devices/ascend/tensor_desc.h"
#include "../../../operators.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm.h"
#include <algorithm>

struct RMSNormAclnnDescriptor {
    Device device;
    aclnnTensorDesc_t xDesc, wDesc, yDesc, rstdDesc;

    RMSNormAclnnDescriptor(Device device);
    void createAclnnDescriptors() {
        aclnnCreateTensorDescriptor(&xDesc);
        aclnnCreateTensorDescriptor(&wDesc);
        aclnnCreateTensorDescriptor(&yDesc);
        aclnnCreateTensorDescriptor(&rstdDesc);
    }
    void destroyAclnnDescriptors() {
        aclnnDestoryTensorDescriptor(xDesc);
        aclnnDestoryTensorDescriptor(wDesc);
        aclnnDestoryTensorDescriptor(yDesc);
        aclnnDestoryTensorDescriptor(rstdDesc);
    }
    void setRstdDescriptor();
};

void rms_norm_aclnn_f16(RMSNormAclnnDescriptor *descriptor, Tensor y, Tensor x,
                        Tensor w, float epsilon, void *stream);

#endif
