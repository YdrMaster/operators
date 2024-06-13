#include "matmul_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "../blas.h"
#include <cmath>

void matmul_cpu_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha) {
    auto a_matrix = BlasMatrix(a.layout, a.data);
    auto b_matrix = BlasMatrix(b.layout, b.data);
    auto c_matrix = BlasMatrix(c.layout, c.data);

    ASSERT_EQ(c_matrix.rows, a_matrix.rows);// m
    ASSERT_EQ(c_matrix.cols, b_matrix.cols);// n
    ASSERT_EQ(a_matrix.cols, b_matrix.rows);// k

    auto batch = c_matrix.batch;
    ASSERT_EQ(batch, a_matrix.batch);
    ASSERT_EQ(batch, b_matrix.batch);

    auto m = c_matrix.rows;
    auto n = c_matrix.cols;
    auto k = a_matrix.cols;

    for (int i = 0; i < batch; ++i) {
        for (int m_ = 0; m_ < m; ++m_) {
            for (int n_ = 0; n_ < n; ++n_) {
                auto c_ptr = reinterpret_cast<uint16_t *>(c.data) + i * m * n + m_ * m;
                auto a_ptr = reinterpret_cast<uint16_t const *>(a_matrix.data) + i * a_matrix.stride + m_ * a_matrix.row_stride;
                auto b_ptr = reinterpret_cast<uint16_t const *>(b_matrix.data) + i * b_matrix.stride + n_ * b_matrix.col_stride;
                for (int k_ = 0; k_ < k; ++k_) {
                    auto a_ = f16_to_f32(a_ptr[k_]);
                    auto b_ = f16_to_f32(b_ptr[k_]);
                    c_ptr[n_] = f32_to_f16(beta * f16_to_f32(c_ptr[n_]) + alpha * a_ * b_);
                }
            }
        }
    }
}
