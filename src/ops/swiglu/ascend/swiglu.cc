#include "swiglu.h"
#include "aclnnop/aclnn_cat.h"
#include "aclnnop/aclnn_swi_glu.h"
#include "swiglu_meta.h"
#include <vector>

extern "C" void swiglu_kernel_do(void *x, void *y, SwiGLUMetaData meta_data,
                                 void *stream);

SwigluAscendCDescriptor::SwigluAscendCDescriptor(Device device) {
    this->device = device;
}

/// @brief  SwiGLU with ascendc impl
/// @param descriptor Tensor meta data
/// @param gate up * swish(gate)
/// @param up
/// @param stream Processor stream
void swiglu_ascendc(SwigluAscendCDescriptor *descriptor, Tensor gate, Tensor up,
                    void *stream) {
    // Check input shape
    ASSERT_EQ(gate.layout->ndim, 2);
    ASSERT_EQ(up.layout->ndim, 2);
    ASSERT_EQ(gate.layout->shape[0], up.layout->shape[0]);
    ASSERT_EQ(gate.layout->shape[1], up.layout->shape[1]);

    auto gateDesc = descriptor->out;
    auto upDesc = descriptor->in;

    // Set aclnnTensorDescriptor from Tensor.layout
    aclnnSetTensorDescriptorFromTensorLayout(gateDesc, gate.layout);
    aclnnSetTensorDescriptorFromTensorLayout(upDesc, up.layout);

    char *descStra = aclnnTensorDescToString(gateDesc);
    if (descStra) {
        printf("%s", descStra);
    } else {
        printf("Failed to print.\n");
    }

    descStra = aclnnTensorDescToString(upDesc);
    if (descStra) {
        printf("%s", descStra);
    } else {
        printf("Failed to print.\n");
    }

    auto gateLen = numElements(gateDesc->shape, gateDesc->ndim);
    auto upLen = numElements(upDesc->shape, upDesc->ndim);
    ASSERT_EQ(gateLen, upLen);

    // Set tileLen as a line of tensor
    auto tileLen = gateDesc->shape[1];

    SwiGLUMetaData swiglu_meta_data = {static_cast<int>(gateLen),
                                       static_cast<int>(tileLen),
                                       static_cast<int>(upDesc->stride[0]),
                                       static_cast<int>(gateDesc->stride[0]),
                                       1.0,
                                       gateDesc->dataType};

    swiglu_kernel_do(up.data, gate.data, swiglu_meta_data, stream);

    return;
}

void swiglu_aclnn_f16(SwigluAscendCDescriptor *descriptor, Tensor gate,
                      Tensor up, void *stream) {
    // Rename in/out tensor descriptor
    auto gateDesc = descriptor->out;
    auto upDesc = descriptor->in;
    aclnnTensorDesc_t catDesc;

    // Copy tensor layout to descriptor
    aclnnSetTensorDescriptorFromTensorLayout(gateDesc, gate.layout);
    aclnnSetTensorDescriptorFromTensorLayout(upDesc, up.layout);

    // Create aclnnCat out tensor
    aclnnCreateTensorDescriptor(&catDesc);
    std::vector<int64_t> catShape(gateDesc->ndim, 1);
    for (auto i = 0; i < gateDesc->ndim; ++i) {
        if (i != 1) {
            ASSERT_EQ(gateDesc->shape[i], upDesc->shape[i]);
            catShape[i] = gateDesc->shape[i];
        }
    }
    catShape[1] = gateDesc->shape[1] + upDesc->shape[1];
    aclnnSetTensorDescriptor(catDesc, catShape.data(), nullptr, catShape.size(),
                             0, gateDesc->dataType, gateDesc->format);

    // Malloc catDesc data
    void *cat_data = nullptr;
    auto ret =
        aclrtMalloc(&cat_data, numElements(catDesc->shape, catDesc->ndim),
                    ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));

    // aclnnCreateTensor
    aclTensor *tgate;
    aclTensor *tup;
    aclTensor *tcat;

    aclnnCreateTensor(gateDesc, gate.data, &tgate);
    aclnnCreateTensor(upDesc, up.data, &tup);
    aclnnCreateTensor(catDesc, cat_data, &tcat);

    // Prepare aclnnCat
    std::vector<aclTensor *> tmp{tgate, tup};
    aclTensorList *tensorList = aclCreateTensorList(tmp.data(), tmp.size());
    // aclnnCat
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    ret = aclnnCatGetWorkspaceSize(tensorList, -1, tcat, &workspaceSize,
                                   &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCatGetWorkspaceSize failed. ERROR: %d\n", ret));
    // Malloc workspace on device
    void *workspaceAddr = mallocWorkspace(workspaceSize);
    ret = aclnnCat(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnCat failed. ERROR: %d\n", ret));
    freeWorkspace(workspaceAddr);

    // Pretare aclnnSwiGlu
    workspaceSize = 0;
    executor = nullptr;
    ret =
        aclnnSwiGluGetWorkspaceSize(tcat, -1, tgate, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret));
    workspaceAddr = mallocWorkspace(workspaceSize);
    ret = aclnnSwiGlu(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSwiGlu failed. ERROR: %d\n", ret));

    // Wait device work
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret));

    aclDestroyTensorList(tensorList);
    aclDestroyTensor(tcat);

    aclrtFree(cat_data);
    aclnnDestoryTensorDescriptor(catDesc);
}
