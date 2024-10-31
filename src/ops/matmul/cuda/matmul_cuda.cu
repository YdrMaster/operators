#include "../../../devices/cuda/cuda_handle.h"
#include "../../utils.h"
#include "../blas.h"
#include "matmul_cuda.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

template<typename Tdata>
infiniopStatus_t matmul_cuda(MatmulCudaDescriptor_t desc, void *c, float beta, void const *a, void const *b, float alpha, void *stream) {
    auto info = desc->info;

    if (info.is_transed) {
        std::swap(a, b);
    }

    Tdata alpha_, beta_;
    cudaDataType a_type, b_type, c_type;
    cublasComputeType_t compute_type;

    if constexpr (std::is_same<Tdata, half>::value) {
        alpha_ = __float2half(alpha);
        beta_ = __float2half(beta);
        a_type = b_type = c_type = CUDA_R_16F;
        compute_type = CUBLAS_COMPUTE_16F;
    } else {
        alpha_ = alpha;
        beta_ = beta;
        a_type = b_type = c_type = CUDA_R_32F;
        compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    }

    auto op_a = info.a_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;
    auto op_b = info.b_matrix.row_stride == 1 ? CUBLAS_OP_N : CUBLAS_OP_T;

    use_cublas(desc->cublas_handles_t, desc->device_id, (cudaStream_t) stream,
               [&](cublasHandle_t handle) { cublasGemmStridedBatchedEx(
                                                handle,
                                                op_a,
                                                op_b,
                                                info.m,
                                                info.n,
                                                info.k,
                                                &alpha_,
                                                a,
                                                a_type,
                                                info.a_matrix.ld(),
                                                info.a_matrix.stride,
                                                b,
                                                b_type,
                                                info.b_matrix.ld(),
                                                info.b_matrix.stride,
                                                &beta_,
                                                c,
                                                c_type,
                                                info.c_matrix.ld(),
                                                info.c_matrix.stride,
                                                info.batch,
                                                compute_type,
                                                CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    cudaDeviceSynchronize();
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaMatmul(MatmulCudaDescriptor_t desc,
                            void *workspace,
                            uint64_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            void *stream) {
    if (desc->dtype == F16) {
        return matmul_cuda<half>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    if (desc->dtype == F32) {
        return matmul_cuda<float>(desc, c, desc->beta, a, b, desc->alpha, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}