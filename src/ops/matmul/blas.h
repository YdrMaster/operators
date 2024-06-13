#ifndef __BLAS_H__
#define __BLAS_H__

#include "../../operators.h"
#include <stdint.h>

struct BlasMatrix {
    int batch;
    int64_t stride;
    int rows;
    int cols;
    int row_stride;
    int col_stride;
    void const *data;

    BlasMatrix(TensorLayout layout, void const *data) {
        if (layout.ndim == 2) {
            this->batch = 1;
            this->stride = 0;
            this->rows = layout.shape[0];
            this->cols = layout.shape[1];
            this->row_stride = layout.strides[0] / layout.dt.size;
            this->col_stride = layout.strides[1] / layout.dt.size;
            this->data = data;
        } else if (layout.ndim == 3) {
            this->batch = layout.shape[0];
            this->stride = layout.strides[0] / layout.dt.size;
            this->rows = layout.shape[1];
            this->cols = layout.shape[2];
            this->row_stride = layout.strides[1] / layout.dt.size;
            this->col_stride = layout.strides[2] / layout.dt.size;
            this->data = data;
        } else {
            PANIC(InvalidMatrixShape);
        }

        if (this->row_stride != 1 && this->col_stride != 1) {
            ASSERT(false);
            PANIC(MatrixIsNotContiguous);
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
