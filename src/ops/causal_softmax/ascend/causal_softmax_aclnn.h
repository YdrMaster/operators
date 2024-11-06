#ifndef __ACLNN_CAUSAL_SOFTMAX_H__
#define __ACLNN_CAUSAL_SOFTMAX_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "operators.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_masked_softmax_with_rel_pos_bias.h>

struct CausalSoftmaxAclnnDescriptor {
    Device device;
    int device_id;
    aclOpExecutor *executor;
    aclnnTensorDescriptor_t aDesc, maskDesc, outDesc;
    uint64_t workspaceSize;
    void *maskAddr;

    CausalSoftmaxAclnnDescriptor(Device device);
};

typedef CausalSoftmaxAclnnDescriptor *CausalSoftmaxAclnnDescriptor_t;

infiniopStatus_t aclnnCreateCausalSoftmaxDescriptor(AscendHandle_t handle,
                                                    CausalSoftmaxAclnnDescriptor_t *desc_ptr,
                                                    infiniopTensorDescriptor_t y_desc);

infiniopStatus_t aclnnGetCausalSoftmaxWorkspaceSize(CausalSoftmaxAclnnDescriptor_t desc, uint64_t *size);

infiniopStatus_t aclnnCausalSoftmax(CausalSoftmaxAclnnDescriptor_t desc,
                                    void *workspace,
                                    uint64_t workspace_size,
                                    void *data,
                                    void *stream);

infiniopStatus_t aclnnDestroyCausalSoftmaxDescriptor(CausalSoftmaxAclnnDescriptor_t desc);

#endif
