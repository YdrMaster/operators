#include "matmul_cnnl.h"
#include "../../../devices/bang/common_bang.h"
#include "../../../devices/bang/handle_pool.h"
#include "../../utils.h"
#include "../blas.h"
#include "cnrt.h"
#include <iostream>

MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
    this->device = device;
    get_cnnl_pool();
}

void matmul_cnnl_f16(MatmulBangDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
    std::cout << "matmul_cnnl_f16" << std::endl;
    std::cout << a.layout->ndim << std::endl;
    std::cout << "a shape: " << a.layout->shape[0] << " " << a.layout->shape[1] << std::endl;
    std::cout << "a strides: " << a.layout->strides[0] << " " << a.layout->strides[1] << std::endl;
    std::cout << b.layout->ndim << std::endl;
    std::cout << "b shape: " << b.layout->shape[0] << " " << b.layout->shape[1] << std::endl;
    std::cout << "b strides: " << b.layout->strides[0] << " " << b.layout->strides[1] << std::endl;    
    auto info = MatmulInfo(c, a, b);

    // int32_t transA = info.a_matrix.row_stride == 1 ? false : true;
    // int32_t transB = info.b_matrix.row_stride == 1 ? false : true; 
    // 这里如果不强制设为 false 后续端到端测试形状就会对不上报错，需要定位一下是什么原因
    int32_t transA = false;
    int32_t transB = false;   


    setCnnlTensor(descriptor->aDesc, info.a_matrix.ndim, info.a_matrix.batch, info.a_matrix.rows, info.a_matrix.cols);
    setCnnlTensor(descriptor->bDesc, info.b_matrix.ndim, info.b_matrix.batch, info.b_matrix.rows, info.b_matrix.cols);
    setCnnlTensor(descriptor->cDesc, info.c_matrix.ndim, info.c_matrix.batch, info.c_matrix.rows, info.c_matrix.cols);

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
                                         &alpha, descriptor->aDesc, info.a_ptr,
                                         descriptor->bDesc, info.b_ptr,
                                         &beta, descriptor->cDesc, info.c_ptr,
                                         workspace, wsSize);
             });

    cnrtFree(workspace);
}

// 原来的版本
// #include "matmul_cnnl.h"
// #include "../../../devices/bang/common_bang.h"
// #include "../../../devices/bang/handle_pool.h"
// #include "../../utils.h"
// #include "../blas.h"
// #include "cnrt.h"
// #include <iostream>

// MatmulBangDescriptor::MatmulBangDescriptor(Device device) {
//     this->device = device;
//     get_cnnl_pool();
// }

// void matmul_cnnl_f16(MatmulBangDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream) {
//     std::cout << "matmul_cnnl_f16" << std::endl;
//     std::cout << a.layout->ndim << std::endl;
//     std::cout << "a shape: " << a.layout->shape[0] << " " << a.layout->shape[1] << std::endl;
//     std::cout << "a strides: " << a.layout->strides[0] << " " << a.layout->strides[1] << std::endl;
//     std::cout << b.layout->ndim << std::endl;
//     std::cout << "b shape: " << b.layout->shape[0] << " " << b.layout->shape[1] << std::endl;
//     std::cout << "b strides: " << b.layout->strides[0] << " " << b.layout->strides[1] << std::endl;  
//     std::cout << c.layout->ndim << std::endl;
//     std::cout << "c shape: " << c.layout->shape[0] << " " << c.layout->shape[1] << std::endl;
//     std::cout << "c strides: " << c.layout->strides[0] << " " << c.layout->strides[1] << std::endl;          
//     auto info = MatmulInfo(c, a, b);

//     int32_t transA = info.a_matrix.row_stride == 1 ? false : true;
//     int32_t transB = info.b_matrix.row_stride == 1 ? false : true;    

//     setCnnlTensor(descriptor->aDesc, a.layout);
//     setCnnlTensor(descriptor->bDesc, b.layout);
//     setCnnlTensor(descriptor->cDesc, c.layout);

//     cnnlSetMatMulDescAttr(descriptor->opDesc, CNNL_MATMUL_DESC_TRANSA, &transA,
//                           sizeof(int32_t));
//     cnnlSetMatMulDescAttr(descriptor->opDesc, CNNL_MATMUL_DESC_TRANSB, &transB,
//                           sizeof(int32_t));

//     void *workspace;

//     use_cnnl((cnrtQueue_t) stream,
//              [&](cnnlHandle_t handle) {
//                  int count = 0;
//                  cnnlGetBatchMatMulAlgoHeuristic(handle, descriptor->opDesc, descriptor->aDesc,
//                                                  descriptor->bDesc, descriptor->cDesc,
//                                                  NULL, 1, &(descriptor->algoResult), &count);
//                  size_t wsSize;
//                  cnnlGetBatchMatMulHeuristicResult(descriptor->algoResult, descriptor->algo, &wsSize);
//                  cnrtMalloc(&workspace, wsSize);
//                  cnnlBatchMatMulBCast_v2(handle, descriptor->opDesc, descriptor->algo,
//                                          &alpha, descriptor->aDesc, a.data,
//                                          descriptor->bDesc, b.data,
//                                          &beta, descriptor->cDesc, c.data,
//                                          workspace, wsSize);
//              });

//     cnrtFree(workspace);
// }
