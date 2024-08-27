#include "reform.h"
#include "aclnnop/aclnn_copy.h"

ReformAscendDescriptor::ReformAscendDescriptor(Device device) {
    this->device = device;
}

void reform_aclnn(ReformAscendDescriptor *descriptor, Tensor y, Tensor x,
                  void *stream) {

    auto selfDesc = descriptor->self;
    auto srcDesc = descriptor->src;

    aclnnSetTensorDescriptorFromTensorLayout(selfDesc, y.layout);
    aclnnSetTensorDescriptorFromTensorLayout(srcDesc, x.layout);

    // char *descStra = aclnnTensorDescToString(selfDesc);
    // if (descStra) {
    //     printf("%s", descStra);
    // } else {
    //     printf("Failed to print.\n");
    // }


    // descStra = aclnnTensorDescToString(srcDesc);
    // if (descStra) {
    //     printf("%s", descStra);
    // } else {
    //     printf("Failed to print.\n");
    // }


    aclTensor *tself;
    aclTensor *tsrc;

    aclnnCreateTensor(selfDesc, y.data, &tself);
    aclnnCreateTensor(srcDesc, x.data, &tsrc);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto ret = aclnnInplaceCopyGetWorkspaceSize(tself, tsrc, &workspaceSize,
                                                &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret));
    void *workspaceAddr = mallocWorkspace(workspaceSize);
    ret = aclnnInplaceCopy(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret));
    
    // Wait device work
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

    aclDestroyTensor(tself);
    aclDestroyTensor(tsrc);
}