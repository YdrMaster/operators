#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../tensor.h"
#include "cnnl.h"
#include <vector>
#include <iostream>

const int NRAM_MAX_SIZE = 1024 * 256;//the maximum NRAM memory is 1024 * 768
const int GDRAM_MAX_SIZE = 1024 * 1024 * 1024;
inline void setCnnlTensor(cnnlTensorDescriptor_t desc, const TensorLayout* layout) {
    std::vector<int> dims(layout->ndim);
    std::cout << "dims: " << layout->ndim << std::endl;
    for (uint64_t i = 0; i < layout->ndim; i++) {
        dims[i] = static_cast<int>(layout->shape[i]);
    }
}

inline void setCnnlTensor(cnnlTensorDescriptor_t desc, int ndim, int batch, int rows, int cols) {
    // std::vector<int> dims(layout->ndim);
    // for (uint64_t i = 0; i < layout->ndim; i++) {
    //     dims[i] = static_cast<int>(layout->shape[i]);
    // }

    std::cout << "setCnnlTensor: " << batch << " " << rows << " " << cols << std::endl;   
    if (ndim == 3) {
        std::vector<int> dims = {batch, rows, cols};
        cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                dims.size(), dims.data());
    }  else if (ndim == 2){
        std::vector<int> dims = {rows, cols};
        cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                dims.size(), dims.data());        
    }
}

#endif  // __COMMON_BANG_H__
