#include "matmul_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include "../blas.h"
#include <cmath>

void matmul_cpu_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha) {
    auto info = MatmulInfo(c, a, b);

    for (int i = 0; i < info.batch; ++i) {
        for (int m_ = 0; m_ < info.m; ++m_) {
            for (int n_ = 0; n_ < info.n; ++n_) {
                auto c_ = reinterpret_cast<uint16_t *>(info.c_ptr) + i * info.c_matrix.stride + m_ * info.c_matrix.row_stride + n_ * info.c_matrix.col_stride;
                float sum = 0;
                for (int k_ = 0; k_ < info.k; ++k_) {
                    auto a_ = reinterpret_cast<uint16_t const *>(info.a_ptr) + i * info.a_matrix.stride + m_ * info.a_matrix.row_stride + k_ * info.a_matrix.col_stride;
                    auto b_ = reinterpret_cast<uint16_t const *>(info.b_ptr) + i * info.b_matrix.stride + n_ * info.b_matrix.col_stride + k_ * info.b_matrix.row_stride;
                    sum += f16_to_f32(*a_) * f16_to_f32(*b_);
                }
                *c_ = f32_to_f16(beta * f16_to_f32(*c_) + alpha * sum);
            }
        }
    }
}
