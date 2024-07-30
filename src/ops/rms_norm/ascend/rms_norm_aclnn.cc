#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"


RMSNormAclnnDescriptor::RMSNormAclnnDescriptor(Device device) {
    this->device = device;
}

void rms_norm_aclnn_f16(RMSNormAclnnDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    // Copy tensor layout to descriptor
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->xDesc, x.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->wDesc, w.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->yDesc, y.layout);

    // Create rsdOut Descriptor
    // aclnnTensorDesc_t rDesc;
    // aclnnCreateTensorDescriptor(&rDesc);
    // aclnnSetTensorDescriptor(rDesc, )

    aclTensor *x;
    aclTensor *gamma;
    aclTensor *yOut;
    // aclTensor *rstdOut;

    aclnnCreateTensor(descriptor->xDesc, x.data, &x);
    aclnnCreateTensor(descriptor->wDesc, w.data, &gamma);
    aclnnCreateTensor(descriptor->yDesc, y.data, &yOut);
    // aclnnCreate

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // ERROR: rstdOut can not set nullptr
    auto ret =
        aclnnRmsNormGetWorkspaceSize(
            x, gamma, epsilon, yOut, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret));
    // Malloc workpace on device
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    }
    // Call aclnnRmsNorm
    ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, reinterpret_cast<aclrtStream>(stream));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret));

    // Wait device work
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret););

    aclDestroyTensor(x);
    aclDestroyTensor(gamma);
    aclDestroyTensor(yOut);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return;
}
