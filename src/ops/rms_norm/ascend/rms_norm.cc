#include "rms_norm.h"


void rms_norm_ascend_npu_fp16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    // Cast from uint64_t* to int64_t*
    auto xshape = castToInt64_t(x.layout.shape, x.layout.ndim);
    auto wshape = castToInt64_t(w.layout.shape, w.layout.ndim);
    auto yshape = castToInt64_t(y.layout.shape, y.layout.ndim);

    auto input = aclCreateTensor(
        xshape, x.layout.ndim, aclDataType::ACL_FLOAT16, x.layout.strides, 0,
        aclFormat::ACL_FORMAT_ND, xshape, x.layout.ndim, (void *) x.data);

    auto weight = aclCreateTensor(
        wshape, w.layout.ndim, aclDataType::ACL_FLOAT16, w.layout.strides, 0,
        aclFormat::ACL_FORMAT_ND, wshape, w.layout.ndim, (void *) w.data);

    auto output = aclCreateTensor(
        yshape, y.layout.ndim, aclDataType::ACL_FLOAT16, y.layout.strides, 0,
        aclFormat::ACL_FORMAT_ND, yshape, y.layout.ndim, (void *) y.data);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // ERROR: rstdOut can not set nullptr
    auto ret =
        aclnnRmsNormGetWorkspaceSize(
            input, weight, epsilon, output, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNormGetWorkspaceSize failed. ERROR: %d\n", ret));

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    }

    ret = aclnnRmsNorm(workspaceAddr, workspaceSize, executor, reinterpret_cast<aclrtStream>(stream));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRmsNorm failed. ERROR: %d\n", ret));

    return;
}