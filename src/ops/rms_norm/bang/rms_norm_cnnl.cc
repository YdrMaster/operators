#include "rms_norm_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "cnrt.h"

RMSNormBangDescriptor::RMSNormBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void rms_norm_cnnl_f16(RMSNormBangDescriptor *descriptor, Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    ASSERT_EQ(y.layout->ndim, 2);
    ASSERT_EQ(x.layout->ndim, 2);
    ASSERT_EQ(w.layout->ndim, 1);

    auto n = y.layout->shape[0],
         d = y.layout->shape[1];

    ASSERT_EQ(x.layout->shape[0], n);
    ASSERT_EQ(x.layout->shape[1], d);
    ASSERT_EQ(w.layout->shape[0], d);

    setCnnlTensor(descriptor->yDesc, y.layout);
    setCnnlTensor(descriptor->xDesc, x.layout);
    setCnnlTensor(descriptor->wDesc, w.layout);

    cnnlSetFuseNormDescriptor(descriptor->opDesc, epsilon, 1.0, true,
                              false, false, false, false,
                              CNNL_DTYPE_HALF, CNNL_TRANSFORMER_RMSNORM);

    void *workspace;
    
    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 size_t wsSize;
                 cnnlGetFuseNormWorkspaceSize(handle, descriptor->opDesc, descriptor->xDesc, &wsSize);
                 cnrtMalloc(&workspace, wsSize);
                 cnnlFuseNorm(handle, descriptor->opDesc, descriptor->xDesc, x.data,
                              descriptor->wDesc, w.data, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr,
                              workspace, wsSize, descriptor->yDesc, y.data, nullptr, nullptr);
             });

    cnrtFree(workspace);
}
