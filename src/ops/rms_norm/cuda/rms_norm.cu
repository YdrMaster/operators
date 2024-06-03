#include "../../../utils.h"
#include "../../c_interface/cuda/nv_gpu.cuh"
#include "rms_norm.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

// assert BLOCK_SIZE >= blockDim.x
template<unsigned int BLOCK_SIZE, class Tdata>
static __global__ void padding(
    Tdata *__restrict__ o_,
    unsigned int const stride_y,
    Tdata const *__restrict__ x_,
    unsigned int const stride_x,
    Tdata const *__restrict__ w_,
    float const epsilon) {
    auto y = o_ + blockIdx.x * stride_y + threadIdx.x;
    auto x = x_[blockIdx.x * stride_x + threadIdx.x];
    auto w = w_[threadIdx.x];

    using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockOp::TempStorage temp_storage;
    auto acc = BlockOp(temp_storage).Reduce(x * x, cub::Sum());

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(blockDim.x) + epsilon));
    }
    __syncthreads();

    *y = rms * x * w;
}

template<unsigned int BLOCK_SIZE, unsigned int ITEMS_PER_THREAD, class Tdata>
static __global__ void folding(
    Tdata *__restrict__ y,
    unsigned int const stride_y,
    Tdata const *__restrict__ x,
    unsigned int const stride_x,
    Tdata const *__restrict__ w,
    float const epsilon,
    unsigned int const items_size) {
    y += blockIdx.x * stride_y;
    x += blockIdx.x * stride_x;

    float thread_data[ITEMS_PER_THREAD];
    {
        using BlockOp = cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, thread_data, items_size, 0.f);
    }

    float squared[ITEMS_PER_THREAD];
#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        squared[i] = thread_data[i] * thread_data[i];
    }

    float acc;
    {
        using BlockOp = cub::BlockReduce<float, BLOCK_SIZE>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        acc = BlockOp(temp_storage).Reduce(squared, cub::Sum());
    }

    __shared__ Tdata rms;
    if (threadIdx.x == 0) {
        rms = Tdata(rsqrtf(acc / float(items_size) + epsilon));
    }
    __syncthreads();

#pragma unroll
    for (unsigned int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (auto j = i + threadIdx.x * ITEMS_PER_THREAD; j < items_size) {
            y[j] = Tdata(float(rms) * float(thread_data[i]) * float(w[j]));
        }
    }
}

constexpr static int
    HIDDEN_SIZE = 4096,
    BLOCK_SIZE = 1024,
    ITEMS_PER_THREAD = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

void rms_norm_nv_gpu_f16(Kernel const *kn, MutTensor y, ConstTensor x, ConstTensor w, float epsilon) {
    ASSERT_EQ(y.layout.ndim, 2);
    ASSERT_EQ(x.layout.ndim, 2);
    ASSERT_EQ(w.layout.ndim, 1);

    auto n = y.layout.shape[0],
         d = y.layout.shape[1];

    ASSERT_EQ(x.layout.shape[0], n);
    ASSERT_EQ(x.layout.shape[1], d);
    ASSERT_EQ(w.layout.shape[0], d);

    auto y_ = reinterpret_cast<half *>(y.data);
    auto x_ = reinterpret_cast<half const *>(x.data);
    auto w_ = reinterpret_cast<half const *>(w.data);

    auto stride_y = y.layout.pattern[0];
    auto stride_x = x.layout.pattern[0];

    auto stream = reinterpret_cast<NvGpuRtCtx const *>(kn->rt_ctx)->stream;

    if (d <= BLOCK_SIZE) {
        padding<BLOCK_SIZE>
            <<<n, d, 0, stream>>>(y_, stride_y, x_, stride_x, w_, epsilon);
    } else {
        folding<BLOCK_SIZE, ITEMS_PER_THREAD>
            <<<n, BLOCK_SIZE, 0, stream>>>(y_, stride_y, x_, stride_x, w_, epsilon, d);
    }
}
