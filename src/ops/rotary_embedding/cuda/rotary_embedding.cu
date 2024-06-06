#include "../../../utils.h"
#include "../../c_interface/cuda/nv_gpu.cuh"
#include "rotary_embedding.cuh"
#include <cuda_fp16.h>

static __global__ void padding(
    half2 *__restrict__ x_,
    unsigned int const *__restrict__ pos_,
    float const theta,
    unsigned int const leading_dim) {
    auto dh = blockDim.x;
    auto k = threadIdx.x;

    auto &x = x_[blockIdx.x * leading_dim + blockIdx.y * dh + k];
    auto pos = float(pos_[blockIdx.x]);

    float sin, cos;
    sincosf(pos / powf(theta, float(k) / float(dh)), &sin, &cos);

    x = x * half2(cos, cos) + half2(-x.y, x.x) * half2(sin, sin);
}

constexpr static int
    BLOCK_SIZE = 1024;

void rotary_embedding_nv_gpu_f16(struct Kernel const *kn, MutTensor t, ConstTensor pos, float theta) {
    ASSERT_EQ(t.layout.ndim, 3);
    ASSERT_EQ(pos.layout.ndim, 1);

    auto nt = t.layout.shape[0],
         nh = t.layout.shape[1],
         dh = t.layout.shape[2];

    ASSERT_EQ(pos.layout.shape[0], nt);
    ASSERT(dh < BLOCK_SIZE);

    auto t_ptr = reinterpret_cast<half2 *>(t.data);
    auto pos_ptr = reinterpret_cast<unsigned int const *>(pos.data);
    auto leading_dim = t.layout.strides[0] / 4;

    auto stream = reinterpret_cast<NvGpuRtCtx const *>(kn->rt_ctx)->stream;
    padding<<<dim3(nt, nh), dh / 2, 0, stream>>>(t_ptr, pos_ptr, theta, leading_dim);
}
