#ifndef __ACLNN_REFORM_H__
#define __ACLNN_REFORM_H__

#include "../../../devices/ascend/common_ascend.h"
#include "../../../devices/ascend/tensor_desc.h"
#include "../../../operators.h"
#include "../../utils.h"

struct ReformAscendDescriptor {
    Device device;
    aclnnTensorDesc_t self, src;

    ReformAscendDescriptor(Device device);
    void createAclnnDescriptors() {
        aclnnCreateTensorDescriptor(&self);
        aclnnCreateTensorDescriptor(&src);
    }
    void destroyAclnnDescriptors() {
        aclnnDestoryTensorDescriptor(self);
        aclnnDestoryTensorDescriptor(src);
    }
};

void reform_aclnn(ReformAscendDescriptor *descriptor, Tensor y, Tensor x,
                  void *stream);

#endif