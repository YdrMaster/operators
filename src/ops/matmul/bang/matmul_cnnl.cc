#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "../blas.h"
#include "cnrt.h"

MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void matmul_cnnl_f16(MatmulBangDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    auto info = MatmulInfo(c, a, b);

    int32_t transA = info.a_matrix.row_stride == 1 ? false : true;
    int32_t transB = info.b_matrix.row_stride == 1 ? false : true;

    setCnnlTensor(descriptor->aDesc, a.layout);
    setCnnlTensor(descriptor->bDesc, b.layout);
    setCnnlTensor(descriptor->cDesc, c.layout);

    cnnlSetMatMulDescAttr(descriptor->opDesc, CNNL_MATMUL_DESC_TRANSA, &transA,
                          sizeof(int32_t));
    cnnlSetMatMulDescAttr(descriptor->opDesc, CNNL_MATMUL_DESC_TRANSB, &transB,
                          sizeof(int32_t));

    void *workspace;

    use_cnnl((cnrtQueue_t) stream,
             [&](cnnlHandle_t handle) {
                 int count = 0;
                 cnnlGetBatchMatMulAlgoHeuristic(handle, descriptor->opDesc, descriptor->aDesc,
                                                 descriptor->bDesc, descriptor->cDesc,
                                                 NULL, 1, &(descriptor->algoResult), &count);
                 size_t wsSize;
                 cnnlGetBatchMatMulHeuristicResult(descriptor->algoResult, descriptor->algo, &wsSize);
                 cnrtMalloc(&workspace, wsSize);
                 cnnlBatchMatMulBCast_v2(handle, descriptor->opDesc, descriptor->algo,
                                         &alpha, descriptor->aDesc, a.data,
                                         descriptor->bDesc, b.data,
                                         &beta, descriptor->cDesc, c.data,
                                         workspace, wsSize);
             });

    cnrtFree(workspace);
}
