#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__

#include "../../../operators.h"
#include "cnnl.h"
#include "cnnl_extra.h"

struct MatmulBangDescriptor {
    Device device;
    cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;

    MatmulBangDescriptor(Device device);
    void createCnnlDescriptors() {
        cnnlCreateTensorDescriptor(&aDesc);
        cnnlCreateTensorDescriptor(&bDesc);
        cnnlCreateTensorDescriptor(&cDesc);
        cnnlMatMulDescCreate(&opDesc);
        cnnlMatMulAlgoCreate(&algo);
        cnnlCreateMatMulHeuristicResult(&algoResult);
    }
    void destroyCnnlDescriptors() {
        cnnlMatMulDescDestroy(opDesc);
        cnnlMatMulAlgoDestroy(algo);
        cnnlDestroyMatMulHeuristicResult(algoResult);
        cnnlDestroyTensorDescriptor(aDesc);
        cnnlDestroyTensorDescriptor(bDesc);
        cnnlDestroyTensorDescriptor(cDesc);
    }
};

void matmul_cnnl_f16(MatmulBangDescriptor *descriptor, Tensor c, float beta, Tensor a, Tensor b, float alpha, void *stream);

#endif// __CNNL_MATMUL_H__
