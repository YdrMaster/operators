#ifndef __ACLNN_MATMUL_H__
#define __ACLNN_MATMUL_H__

#include "../../../devices/ascend/tensor_desc.h"
#include "../../../operators.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_matmul.h"


struct MatmulAclnnDescriptor {
    Device device;
    aclnnTensorDesc_t aDesc, bDesc, cDesc;

    MatmulAclnnDescriptor(Device device);
    void createAclnnDescriptors() {
        aclnnCreateTensorDescriptor(&aDesc);
        aclnnCreateTensorDescriptor(&bDesc);
        aclnnCreateTensorDescriptor(&cDesc);
    }
    void destroyAclnnDescriptors() {
        aclnnDestoryTensorDescriptor(aDesc);
        aclnnDestoryTensorDescriptor(bDesc);
        aclnnDestoryTensorDescriptor(cDesc);
    }
};

void matmul_aclnn_f16(MatmulAclnnDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream);

#endif