#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../devices/ascend/common_ascend.h"
#include "../../../devices/ascend/tensor_desc.h"
#include "../../../operators.h"
#include "../../utils.h"

struct SwigluAscendCDescriptor {
    Device device;
    aclnnTensorDesc_t in, out;

    SwigluAscendCDescriptor(Device device);
    void createAclnnDescriptors() {
        aclnnCreateTensorDescriptor(&in);
        aclnnCreateTensorDescriptor(&out);
    }
    void destroyAclnnDescriptors() {
        aclnnDestoryTensorDescriptor(in);
        aclnnDestoryTensorDescriptor(out);
    }
};


void swiglu_ascendc(SwigluAscendCDescriptor *descriptor, Tensor gate, Tensor up, void *stream);
void swiglu_aclnn_f16(SwigluAscendCDescriptor *descriptor, Tensor gate, Tensor up, void *stream);

#endif