#include "../../utils.h"
#include "../blas.h"
#include "matmul_cuda.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

void matmul_nv_gpu_f16(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream) {
    auto a_matrix = BlasMatrix(a.layout, a.data);
    auto b_matrix = BlasMatrix(b.layout, b.data);
    auto c_matrix = BlasMatrix(c.layout, c.data);

    ASSERT_EQ(c_matrix.rows, a_matrix.rows);// m
    ASSERT_EQ(c_matrix.cols, b_matrix.cols);// n
    ASSERT_EQ(a_matrix.cols, b_matrix.rows);// k

    auto batch = c_matrix.batch;
    if (!a_matrix.match_batch(batch) || !b_matrix.match_batch(batch)) {
        ASSERT(false);
    }

    if (c_matrix.row_stride == 1) {
        // Nothing to do
    } else {
        c_matrix.transpose();
        b_matrix.transpose();
        a_matrix.transpose();
    }

    auto alpha_f16 = __float2half(alpha);
    auto beta_f16 = __float2half(beta);

    auto m = c_matrix.rows;
    auto n = c_matrix.cols;
    auto k = a_matrix.cols;

    auto op_a = a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasGemmStridedBatchedEx(
        handle,
        op_a,
        op_b,
        m,
        n,
        k,
        &alpha_f16,
        a.data,
        CUDA_R_16F,
        a_matrix.ld(),
        a_matrix.stride,
        b.data,
        CUDA_R_16F,
        b_matrix.ld(),
        b_matrix.stride,
        &beta_f16,
        c.data,
        CUDA_R_16F,
        c_matrix.ld(),
        c_matrix.stride,
        batch,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
