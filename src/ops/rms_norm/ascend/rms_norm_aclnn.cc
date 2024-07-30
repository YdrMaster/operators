#include "rms_norm_aclnn.h"
#include "../../../devices/ascend/common_ascend.h"

RMSNormAclnnDescriptor::RMSNormAclnnDescriptor(Device device) {
    this->device = device;
}

void RMSNormAclnnDescriptor::setRstdDescriptor() {
    // Cal weight size
    int64_t wSize = 1;
    for (int64_t i = 0; i < wDesc->ndim; i++) {
        wSize *= (wDesc->shape)[i];
    }
    // Get pos
    int64_t xSize = 1;
    int64_t rstd_dims = xDesc->ndim - 1;
    for (int64_t i = xDesc->ndim - 1; i >= 0; --i) {
        xSize *= (xDesc->shape)[i];
        rstd_dims = i;
        if (xSize == wSize) {
            break;
        }
    }
    auto rstdShape = new int64_t[xDesc->ndim];
    std::fill_n(rstdShape, xDesc->ndim, 1);
    for (int64_t i = 0; i < rstd_dims; i++) {
        rstdShape[i] = (xDesc->shape)[i];
    }
    aclnnSetTensorDescriptor(rstdDesc, rstdShape, nullptr, xDesc->ndim, 0,
                             aclDataType::ACL_FLOAT, aclFormat::ACL_FORMAT_ND);
}

void rms_norm_aclnn_f16(RMSNormAclnnDescriptor *descriptor, Tensor y, Tensor x,
                        Tensor w, float epsilon, void *stream) {
    // Copy tensor layout to descriptor
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->xDesc, x.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->wDesc, w.layout);
    aclnnSetTensorDescriptorFromTensorLayout(descriptor->yDesc, y.layout);

    // Create rsdOut Descriptor
    descriptor->setRstdDescriptor();

    // Malloc rstdOut space on device
    void *rstd_data = nullptr;
    auto rstdDesc = descriptor->rstdDesc;
    auto ret =
        aclrtMalloc(&rstd_data, numElements(rstdDesc->shape, rstdDesc->ndim),
                    ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));

    // aclnnCreateTensor
    aclTensor *tx;
    aclTensor *tgamma;
    aclTensor *ty;
    aclTensor *trstd;

    aclnnCreateTensor(descriptor->xDesc, x.data, &tx);
    aclnnCreateTensor(descriptor->wDesc, w.data, &tgamma);
    aclnnCreateTensor(descriptor->yDesc, y.data, &ty);
    aclnnCreateTensor(descriptor->rstdDesc, rstd_data, &trstd);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // ERROR: rstdOut can not set nullptr
    ret = aclnnRmsNormGetWorkspaceSize(tx, tgamma, epsilon, ty, trstd,
                                       &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret));
    // Malloc workpace on device
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    }
    // Call aclnnRmsNorm
    ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor,
                       reinterpret_cast<aclrtStream>(stream));
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret));

    // Wait device work
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret););

    aclDestroyTensor(tx);
    aclDestroyTensor(tgamma);
    aclDestroyTensor(ty);
    aclDestroyTensor(trstd);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    return;
}
