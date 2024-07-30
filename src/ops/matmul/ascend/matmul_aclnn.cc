#include "matmul_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"
#include "../../utils.h"
#include "../blas.h"

MatmulAclnnDescriptor::MatmulAclnnDescriptor(Device device) {
    this->device = device;
}

void matmul_aclnn_f16(MatmulAclnnDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    // Copy tensor layout to descriptor
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->aDesc, a.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->bDesc, b.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->cDesc, c.layout);

    // char *descStra = aclnnTensorDescToString(descriptor->aDesc);
    // if (descStra) {
    //     printf("%s", descStra);
    // } else {
    //     printf("Failed to print.\n");
    // }

    // char *descStrb = aclnnTensorDescToString(descriptor->bDesc);
    // if (descStrb) {
    //     printf("%s", descStrb);
    // } else {
    //     printf("Failed to print.\n");
    // }

    // char *descStrc = aclnnTensorDescToString(descriptor->cDesc);
    // if (descStrc) {
    //     printf("%s", descStrc);
    // } else {
    //     printf("Failed to print.\n");
    // }

    // Create aclTensor
    aclTensor *self;
    aclTensor *other;
    aclTensor *out;

    aclnnCreateTensor(descriptor->aDesc, a.data, &self);
    aclnnCreateTensor(descriptor->bDesc, b.data, &other);
    aclnnCreateTensor(descriptor->cDesc, c.data, &out);

    // Get workspaceSize
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // TODO
    int8_t cubeMathType = 1;
    auto ret = aclnnBatchMatMulGetWorkspaceSize(
        self, other, out, cubeMathType, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret););
    // Malloc workspace on device
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret););
    }
    // Call aclnnBatchMatMul
    ret = aclnnBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMul failed. ERROR: %d\n", ret););
    
    // Wait device work
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret););

    aclDestroyTensor(self);
    aclDestroyTensor(other);
    aclDestroyTensor(out);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return;
}
