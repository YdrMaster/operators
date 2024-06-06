#include "../../../utils.h"
#include "../../c_interface/cuda/nv_gpu.cuh"
#include "swiglu.cuh"
#include <cuda_fp16.h>

static __forceinline__ __device__ float sigmoid(float x) {
    return fdividef(1, 1 + expf(-x));
}

inline int gcd(int a, int b) {
    while (b != 0) {
        int rem = a % b;
        a = b;
        b = rem;
    }
    return a;
}

template<class Tdata>
static __global__ void swiglu(
    Tdata *__restrict__ gate_,
    int const stride_gate,
    Tdata const *__restrict__ up_,
    int const stride_up) {
    auto i = blockIdx.y * stride_gate + blockIdx.x * blockDim.x + threadIdx.x,
         j = blockIdx.y * stride_up + blockIdx.x * blockDim.x + threadIdx.x;
    auto x = float(gate_[i]),
         y = float(up_[j]);
    gate_[i] = Tdata(x * sigmoid(x) * y);
}

constexpr static int BLOCK_SIZE = 1024;

void swiglu_nv_gpu_f16(struct Kernel const *kn, MutTensor gate, ConstTensor up) {
    ASSERT_EQ(gate.layout.ndim, 2);
    ASSERT_EQ(up.layout.ndim, 2);
    ASSERT_EQ(gate.layout.shape[0], up.layout.shape[0]);
    ASSERT_EQ(gate.layout.shape[1], up.layout.shape[1]);

    auto seq_len = gate.layout.shape[0],
         di = gate.layout.shape[1];

    dim3 block_dims = gcd(BLOCK_SIZE, di);
    dim3 grid_dims = dim3(seq_len, di / block_dims.x);

    auto gate_ptr = reinterpret_cast<half *>(gate.data);
    auto up_ptr = reinterpret_cast<half const *>(up.data);

    auto stream = reinterpret_cast<NvGpuRtCtx const *>(kn->rt_ctx)->stream;

    swiglu<<<grid_dims, block_dims, 0, stream>>>(
        gate_ptr, gate.layout.strides[0], up_ptr, up.layout.strides[0]);
}
