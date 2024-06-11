#ifndef __BLAS_H__
#define __BLAS_H__

#include "../../c_interface/cuda/nv_gpu.cuh"
#include <stdint.h>

struct BlasMatrix {
    int batch;
    int64_t stride;
    int rows;
    int cols;
    int row_stride;
    int col_stride;
    void *data;

    BlasMatrix(TensorLayout layout, void const *data) {
        if (layout.ndim == 2) {
            return BlasMatrix(1,
                              0,
                              layout.shape[0],
                              layout.shape[1],
                              layout.strides[0] / layout.dt.size,
                              layout.strides[1] / layout.dt.size,
                              data);
        } else if (layout.ndim == 3) {
            return BlasMatrix(layout.shape[0],
                              layout.strides[0] / layout.dt.size,
                              layout.shape[1], layout.shape[2],
                              layout.strides[1] / layout.dt.size,
                              layout.strides[2] / layout.dt.size,
                              data);
        } else {
            ASSERT(false);
        }

        if (this->row_stride != 1 && this->col_stride != 1) {
            ASSERT(false);
        }
    }

    bool match_batch(uint64_t batch) const {
        return this->batch == batch || this->batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    int ld() const {
        if (this->row_stride == 1) {
            return this->col_stride;
        } else {
            return this->row_stride;
        }
    }
};

#endif// __BLAS_H__
