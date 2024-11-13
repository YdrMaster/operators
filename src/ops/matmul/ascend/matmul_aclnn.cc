#include "matmul_aclnn.h"

MatmulAclnnDescriptor::MatmulAclnnDescriptor(Device _device) {
    device = _device;
    device_id = 0;
    executor = nullptr;
    info = nullptr;
    cDesc = new aclnnTensorDescriptor();
    aDesc = new aclnnTensorDescriptor();
    bDesc = new aclnnTensorDescriptor();
    alpha = 1.0;
    beta = 0;
    mt = 1;
    workspaceSize = 0;
}

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             float alpha,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             float beta,
                                             int8_t mt) {
    if (c_desc->ndim == 3 && (alpha != 1.0 || beta != 0)) {
        return STATUS_BAD_PARAM;
    }

    *desc_ptr = new MatmulAclnnDescriptor(handle->device);
    (*desc_ptr)->device_id = handle->device_id;
    (*desc_ptr)->mt = mt;
    (*desc_ptr)->alpha = alpha;
    (*desc_ptr)->beta = beta;

    infiniopStatus_t *status = new infiniopStatus_t{STATUS_EXECUTION_FAILED};
    auto info_ptr = new MatmulInfo(c_desc, a_desc, b_desc, status);
    if (*status != STATUS_SUCCESS) {
        return *status;
    }
    (*desc_ptr)->info = info_ptr;

    auto &cDesc = (*desc_ptr)->cDesc;
    auto &aDesc = (*desc_ptr)->aDesc;
    auto &bDesc = (*desc_ptr)->bDesc;

    CHECK_STATUS(cDesc->fromInfiniOpTensorDescriptor(c_desc), STATUS_SUCCESS);
    CHECK_STATUS(aDesc->fromInfiniOpTensorDescriptor(a_desc), STATUS_SUCCESS);
    CHECK_STATUS(bDesc->fromInfiniOpTensorDescriptor(b_desc), STATUS_SUCCESS);

    CHECK_STATUS(cDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(aDesc->createTensor(), STATUS_SUCCESS);
    CHECK_STATUS(bDesc->createTensor(), STATUS_SUCCESS);

    auto b = (*desc_ptr)->info->batch;
    auto &workspaceSize = (*desc_ptr)->workspaceSize;
    auto &executor = (*desc_ptr)->executor;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    aclnnStatus ret;

    if (b > 1) {
        // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnMatmul.md
        ret = aclnnMatmulGetWorkspaceSize(ta,
                                          tb,
                                          tc,
                                          (*desc_ptr)->mt,
                                          &workspaceSize,
                                          &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnMatmulGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return STATUS_EXECUTION_FAILED);
        aclSetAclOpExecutorRepeatable(executor);
    } else {
        // Get transA and transB according strides
        // int64_t transA = aDesc->strides[aDesc->ndim - 1] == 1 ? 0 : 1;
        // int64_t transB = bDesc->strides[bDesc->ndim - 1] == 1 ? 0 : 1;
        int64_t transA = 0;
        int64_t transB = 0;
        // aclnnGemm support C = alpha * A @ B + beta * C
        // see https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/apiref/aolapi/context/aclnnGemm.md
        ret = aclnnGemmGetWorkspaceSize(ta, tb, tc, (*desc_ptr)->alpha, (*desc_ptr)->beta, transA, transB, tc,
                                        (*desc_ptr)->mt, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret);
                  return STATUS_EXECUTION_FAILED);
        aclSetAclOpExecutorRepeatable(executor);
    }

    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             uint64_t *size) {
    *size = desc->workspaceSize;
    return STATUS_SUCCESS;
}

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             void const *a,
                             void const *b,
                             void *stream) {
    auto &cDesc = desc->cDesc;
    auto &aDesc = desc->aDesc;
    auto &bDesc = desc->bDesc;

    aclTensor *tc = cDesc->t;
    aclTensor *ta = aDesc->t;
    aclTensor *tb = bDesc->t;

    auto batch = desc->info->batch;

    auto &executor = desc->executor;
    auto &workspaceSize = desc->workspaceSize;

    // Set runing on handle device
    aclrtSetDevice(desc->device_id);

    aclnnStatus ret;
    if (batch > 1) {
        AclSetTensorAddr(executor, 0, ta, (void *) a);
        AclSetTensorAddr(executor, 1, tb, (void *) b);
        AclSetTensorAddr(executor, 2, tc, (void *) c);
        ret = aclnnMatmul(workspace, workspaceSize, executor, stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnMatmul failed. ERROR: %d\n", ret);
                  return STATUS_EXECUTION_FAILED);
    } else {
        AclSetTensorAddr(executor, 0, ta, (void *) a);
        AclSetTensorAddr(executor, 1, tb, (void *) b);
        AclSetTensorAddr(executor, 2, tc, (void *) c);
        AclSetTensorAddr(executor, 3, tc, (void *) c);
        ret = aclnnGemm(workspace,
                        workspaceSize,
                        executor,
                        stream);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret);
                  return STATUS_EXECUTION_FAILED);
    }

    return STATUS_SUCCESS;
}


infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc) {
    delete desc->cDesc;
    delete desc->bDesc;
    delete desc->aDesc;
    delete desc->info;
    aclDestroyAclOpExecutor(desc->executor);
    delete desc;

    return STATUS_SUCCESS;
}
