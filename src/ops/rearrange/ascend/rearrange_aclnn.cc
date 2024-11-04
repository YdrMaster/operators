#include "rearrange_aclnn.h"
#include "../../utils.h"

RearrangeAclnnDescriptor::RearrangeAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    dstDesc = new aclnnTensorDescriptor();
    srcDesc = new aclnnTensorDescriptor();
    workspaceSize = 0;
    workspaceAddr = nullptr;
}

infiniopStatus_t aclnnCreateRearrangeDescriptor(AscendHandle_t handle,
                                                RearrangeAclnnDescriptor_t *desc_ptr,
                                                infiniopTensorDescriptor_t dst,
                                                infiniopTensorDescriptor_t src) {
    *desc_ptr = new RearrangeAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;

    auto &dstDesc = (*desc_ptr)->dstDesc;
    auto &srcDesc = (*desc_ptr)->srcDesc;

    CHECK_STATUS(dstDesc->fromInfiniOpTensorDescriptor(dst), STATUS_SUCCESS);
    CHECK_STATUS(srcDesc->fromInfiniOpTensorDescriptor(src), STATUS_SUCCESS);

    CHECK_STATUS(dstDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(srcDesc->createTensor(), STATUS_SUCCESS);

    aclTensor *td = dstDesc->t;
    aclTensor *ts = srcDesc->t;

    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;

    auto ret = aclnnInplaceCopyGetWorkspaceSize(td,
                                                ts,
                                                &workspaceSize,
                                                &executor);
    aclSetAclOpExecutorRepeatable(executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplaceCopyGetWorkspaceSize failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);

    (*desc_ptr)->workspaceAddr = mallocWorkspace(workspaceSize);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnRearrange(RearrangeAclnnDescriptor_t desc,
                                void *dst,
                                void const *src,
                                void *stream) {
    // Set runing on handle device
    aclrtSetDevice(desc->device_id);

    aclTensor *td = desc->dstDesc->t;
    aclTensor *ts = desc->srcDesc->t;

    auto &executor = desc->executor;

    AclSetTensorAddr(executor, 0, td, dst);
    AclSetTensorAddr(executor, 1, ts, (void *) src);
    auto ret = aclnnInplaceCopy(desc->workspaceAddr,
                                desc->workspaceSize,
                                executor,
                                stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnInplaceCopy failed. ERROR: %d\n", ret);
              return STATUS_EXECUTION_FAILED);

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnDestroyRearrangeDescriptor(RearrangeAclnnDescriptor_t desc) {
    delete desc->srcDesc;
    delete desc->dstDesc;
    aclDestroyAclOpExecutor(desc->executor);
    freeWorkspace(desc->workspaceAddr);
    delete desc;

    return STATUS_SUCCESS;
}
