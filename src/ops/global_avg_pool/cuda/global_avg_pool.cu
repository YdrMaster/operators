#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "global_avg_pool.cuh"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>

namespace infini {
    struct float2_t {
        float x, y;

        __device__ float2_t() : x(0), y(0) {}
        __device__ float2_t(int val) : x(static_cast<float>(val)), y(static_cast<float>(val)) {}
        __device__ float2_t(const float2 &val) : x(val.x), y(val.y) {}
        __device__ float2_t(const float2_t &other) : x(other.x), y(other.y) {}
        __device__ float2_t(float x, float y) : x(x), y(y) {}

        __device__ float2_t &operator=(const float2_t &other) {
            if (this != &other) {
                this->x = other.x;
                this->y = other.y;
            }
            return *this;
        }

        // __device__ float2 operator=(const int &other) const {
        //     return float2{static_cast<float>(other), static_cast<float>(other)};
        // }

        __device__ float2_t operator+(const float2_t &other) const {
            return float2_t{x + other.x, y + other.y};
        }

        __device__ float operator+(const float &other) const {
            return x + y + other;
        }

        __device__ float2_t &operator+=(const float2_t &other) {
            x += other.x;
            y += other.y;
            return *this;
        }

        __device__ float operator[](size_t index) const {
            return index == 0 ? x : y;
        }
    };

    struct half2 {
        half x, y;

        __device__ half2 &operator=(const half2 &other) {
            if (this != &other) {
                this->x = other.x;
                this->y = other.y;
            }
            return *this;
        }

        __device__ half2 &operator=(const infini::float2_t &other) {
            this->x = __float2half(other.x);
            this->y = __float2half(other.y);
            return *this;
        }

        __device__ half2 operator+(const half2 &other) const {
            return half2{__hadd(x, other.x), __hadd(y, other.y)};
        }

        __device__ half operator+(const half &other) const {
            return __hadd(__hadd(x, y), other);
        }

        __device__ half operator[](size_t index) const {
            return __hadd(x, y);
        }
    };

    struct half4 {
        __half x, y, z, w;

        __device__ half4 operator+(const half4 &other) const {
            return half4{__hadd(x, other.x), __hadd(y, other.y), __hadd(z, other.z), __hadd(w, other.w)};
        }
    };

    __device__ __forceinline__ infini::float2_t divide(infini::float2_t val, float divisor) {
        return {val.x / divisor, val.y / divisor};
    }
}// namespace infini


struct half2float_functor {
    __device__ __forceinline__ float operator()(half val) const {
        return __half2float(val);
    }
};

struct float2half_functor {
    __device__ __forceinline__ half operator()(float val) const {
        return __float2half(val);
    }
};

struct half22float_functor {
    __device__ __forceinline__ float operator()(infini::half2 val) const {
        return __half2float(val.x) + __half2float(val.y);
    }
};

struct float22half2_functor {
    __device__ __forceinline__ infini::half2 operator()(const infini::float2_t &val) const {
        return {__float2half(val.x), __float2half(val.y)};
    }
};

uint64_t getBlockDim(uint64_t size) {
    if (size < static_cast<uint64_t>(MAX_THREADS_PER_BLOCK)) {
        return size;
    }
    for (size_t i = MAX_THREADS_PER_BLOCK; i > 1; --i) {
        if (size % i == 0) {
            return i;
        }
    }
    return 1;
}

/**
 * @brief A templated vector struct that supports element-wise addition on arrays.
 *
 * @tparam T - The access data type for elements in the vector.
 * @tparam TComp - The computation data type used for arithmetic operations.
 * @tparam N - The number of elements of type T in the vector for a single access.
 */
template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];

    __device__ __forceinline__ vecN operator+(const vecN<T, TComp, N> &other) const {
        vecN<T, TComp, N> result;

        for (int i = 0; i < N; ++i) {
            if constexpr (std::is_same<T, TComp>::value) {
                result.data[i] = data[i] + other.data[i];
            } else {
                constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);
                auto data_ = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(result.data);
                data_[i] = std::move(reinterpret_cast<vecN<TComp, TComp, pack_size> const *>(data)[i] +
                                     reinterpret_cast<vecN<TComp, TComp, pack_size> const *>(other.data)[i]);
            }
        }

        return result;
    }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }
};

/** ---------------------------------------- */
/** ---------------   Sum  ----------------- */
/** ---------------------------------------- */

template<typename Tdata, typename TIdata, typename Ldata, int BLOCK_SIZE = 256>
__global__ void sum(
    Ldata *__restrict__ y,
    const Tdata *__restrict__ x,
    uint64_t data_size,
    uint64_t x_per_NC_data_size,
    uint64_t blocks_per_y,
    unsigned remainder,
    uint64_t offset,
    unsigned pack_size) {
    uint64_t block_offset = blockIdx.x / blocks_per_y * x_per_NC_data_size + blockIdx.x % blocks_per_y * blockDim.x * pack_size;
    uint64_t idx = block_offset + threadIdx.x * pack_size + offset;

    if (idx < data_size) {
        Tdata thread_data[1];

        using BlockOp = cub::BlockLoad<Tdata, BLOCK_SIZE, 1, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        __shared__ typename BlockOp::TempStorage load_temp_storage;
        BlockOp(load_temp_storage).Load(x + block_offset, thread_data);

        using BlockReduce = cub::BlockReduce<Tdata, BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage reduce_temp_storage;
        Ldata block_sum;
        if constexpr (std::is_same<Tdata, half>::value) {
            block_sum = BlockReduce(reduce_temp_storage).Sum(__half2float(thread_data[0]), blockDim.x);
        } else {
            block_sum = BlockReduce(reduce_temp_storage).Sum(Ldata(thread_data[0]), blockDim.x);
        }

        // add up the remaining elements
        if (blockIdx.x % blocks_per_y == blocks_per_y - 1) {
            __shared__ typename BlockOp::TempStorage load_r_temp_storage;
            BlockOp(load_r_temp_storage).Load(x + block_offset + blockDim.x, thread_data, remainder, 0);
            if constexpr (std::is_same<Tdata, half>::value) {
                block_sum += __half2float(BlockReduce(reduce_temp_storage).Sum(__half2float(thread_data[0]), blockDim.x));
            } else {
                block_sum += BlockReduce(reduce_temp_storage).Sum(Ldata(thread_data[0]), remainder);
            }
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            atomicAdd(&y[idx / x_per_NC_data_size], block_sum);
        }
    }
}

template<typename Xdata, typename Ydata>
void _sum_nv_gpu(Ydata *y, Xdata const *x, uint64_t data_size, uint64_t x_per_NC_data_size,
                 unsigned int pack_size, uint64_t max_grid_size, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(256);//dim3(std::min(static_cast<uint64_t>(256), x_per_NC_data_size));
    dim3 gridDims = dim3(std::min(data_size / blockDims.x, max_grid_size));
    // uint64_t step = gridDims.x * blockDims.x;
    uint64_t blocks_per_y = x_per_NC_data_size / blockDims.x;
    unsigned int remainder = x_per_NC_data_size % blockDims.x;

    // printf("grid: %d, block: %d\n", gridDims.x, blockDims.x);
    // printf("x_per_NC_data_size: %ld, blocks_per_y: %ld, remainder: %d\n", x_per_NC_data_size, blocks_per_y, remainder);

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    sum<Xdata, Ydata><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, data_size, x_per_NC_data_size, blocks_per_y, remainder, 0, pack_size);
}

template<typename Xdata, typename XIdata, typename Ydata, typename YIdata>
void sum_nv_gpu(void *y, void const *x, uint64_t data_size, uint64_t x_per_NC_data_size, unsigned int pack_size, uint64_t max_grid_size, void *stream) {
    const auto x_ = reinterpret_cast<Xdata const *>(x);
    const auto y_ = reinterpret_cast<Ydata *>(y);
    _sum_nv_gpu<Xdata, Ydata>(y_, x_, data_size, x_per_NC_data_size, pack_size, max_grid_size, stream);
}

/** ---------------------------------------- */
/** --------------   Reset  ---------------- */
/** ---------------------------------------- */
template<typename Tdata>
__global__ void reset(
    Tdata *__restrict__ dst,
    uint64_t data_size,
    uint64_t offset,
    unsigned int pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        dst[idx] = Tdata(0);
    }
}

template<typename Tdata>
void _reset_nv_gpu(Tdata *x, uint64_t data_size, unsigned int pack_size, uint64_t offset, uint64_t max_grid_size, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        reset<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(x, offset + data_size, offset + i, pack_size);
    }
}

template<typename Tdata, typename TIdata>
void reset_nv_gpu(void *x, uint64_t data_size, unsigned int pack_size, uint64_t max_grid_size, void *stream) {
    const auto packed_data_size = data_size / pack_size;
    const auto x_vec = reinterpret_cast<Tdata *>(x);
    _reset_nv_gpu<Tdata>(x_vec, packed_data_size, pack_size, 0, max_grid_size, stream);

    const auto remainder = data_size % pack_size;
    const auto x_ = reinterpret_cast<TIdata *>(x);
    _reset_nv_gpu<TIdata>(x_, remainder, 1, data_size * pack_size, max_grid_size, stream);
}


/** ---------------------------------------- */
/** -------------   Average  --------------- */
/** ---------------------------------------- */
template<typename Xdata, typename Ydata>
__global__ void average(
    Ydata *y,
    Xdata const *x,
    uint64_t data_size,
    uint64_t x_per_NC_data_size,
    uint64_t offset,
    unsigned pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    // printf("idx: %ld, t2l: %ld, %ld, %f\n", idx, T2L(y[idx]), T2L(y[idx]) / data_size, L2T(T2L(y[idx]) / data_size));
    // printf("idx: %ld, size: %f, res: %f\n", idx, static_cast<float>(x_per_NC_data_size), __half2float(__float2half(__half2float(y[idx]) / static_cast<float>(x_per_NC_data_size))));

    if (idx < data_size) {
        // y[idx] = L2T(divide(x[idx], static_cast<Ldata>(x_per_NC_data_size)));
        if constexpr (std::is_same<Xdata, half>::value && std::is_same<Ydata, half>::value) {
            y[idx] = __float2half(__half2float(x[idx]) / x_per_NC_data_size);
        } else if constexpr (std::is_same<Ydata, half>::value) {
            y[idx] = __float2half(x[idx] / x_per_NC_data_size);
        } else if constexpr (std::is_same<Xdata, half>::value) {
            y[idx] = __half2float(x[idx]) / x_per_NC_data_size;
        } else {
            y[idx] = x[idx] / x_per_NC_data_size;
        }
    }
}

template<typename Xdata, typename Ydata>
void _average_nv_gpu(Ydata *y, Xdata const *x, uint64_t data_size, uint64_t x_per_NC_data_size,
                     unsigned int pack_size, uint64_t offset, uint64_t max_grid_size, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        average<Xdata, Ydata><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, offset + data_size, x_per_NC_data_size, offset + i, pack_size);
    }
}

template<typename Xdata, typename XIdata, typename Ydata, typename YIdata>
void average_nv_gpu(void *y, void const *x, uint64_t data_size, uint64_t x_per_NC_data_size, unsigned int pack_size, uint64_t max_grid_size, void *stream) {
    const auto packed_data_size = data_size / pack_size;
    const auto x_vec = reinterpret_cast<Xdata const *>(x);
    const auto y_vec = reinterpret_cast<Ydata *>(y);
    _average_nv_gpu<Xdata, Ydata>(y_vec, x_vec, packed_data_size, x_per_NC_data_size, pack_size, 0, max_grid_size, stream);

    const auto remainder = data_size % pack_size;
    const auto x_ = reinterpret_cast<XIdata const *>(x);
    const auto y_ = reinterpret_cast<YIdata *>(y);
    _average_nv_gpu<XIdata, YIdata>(y_, x_, remainder, x_per_NC_data_size, 1, data_size * pack_size, max_grid_size, stream);
}


/** ---------------------------------------- */
/** ---------   Global Avg Pool  ----------- */
/** ---------------------------------------- */

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata, unsigned int BLOCK_SIZE>
__global__ void global_avg_pool_padding(
    Tdata *__restrict__ y,
    Tdata const *__restrict__ x,
    uint64_t data_size,
    uint64_t x_per_NC_data_size,
    uint64_t offset,
    unsigned pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        Tdata thread_data[1];

        using BlockOp = cub::BlockLoad<Tdata, BLOCK_SIZE, 1, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
        __shared__ typename BlockOp::TempStorage load_temp_storage;
        BlockOp(load_temp_storage).Load(x + blockIdx.x * blockDim.x, thread_data);

        using BlockReduce = cub::BlockReduce<Tdata, BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage reduce_temp_storage;
        Ldata block_sum = BlockReduce(reduce_temp_storage).Sum(Ldata(thread_data[0]), blockDim.x);

        if (threadIdx.x == 0) {
            y[blockIdx.x] = Tdata(block_sum / x_per_NC_data_size);
        }
    }
}

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata>
void launch_global_avg_pool_padding(GlobalAvgPoolCudaDescriptor_t desc, Tdata *y, Tdata const *x, void *stream, unsigned pack_size) {
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(desc->max_block_size), desc->x_per_NC_data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;
    // printf("grid: %d, block: %d\n", gridDims.x, blockDims.x);

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < desc->data_size; i += step) {
        global_avg_pool_padding<Tdata, TIdata, Ldata, LIdata, 256><<<gridDims, blockDims, 0, cuda_stream>>>(
            y, x, desc->data_size, desc->x_per_NC_data_size, i, pack_size);
    }
}


template<typename Tdata, typename TIdata, unsigned int BLOCK_SIZE>
void global_avg_pool_folding_direct(GlobalAvgPoolCudaDescriptor_t desc, void *y, void const *x, void *stream, unsigned pack_size) {
    reset_nv_gpu<Tdata, TIdata>(y, desc->y_data_size, pack_size, desc->max_grid_size, stream);
    sum_nv_gpu<Tdata, TIdata, Tdata, TIdata>(y, x, desc->data_size, desc->x_per_NC_data_size, pack_size, desc->max_grid_size, stream);
    average_nv_gpu<Tdata, TIdata, Tdata, TIdata>(y, y, desc->y_data_size, desc->x_per_NC_data_size, pack_size, desc->max_grid_size, stream);
}

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata, unsigned int BLOCK_SIZE>
void global_avg_pool_folding_workspace(GlobalAvgPoolCudaDescriptor_t desc, void *y, void const *x, void *workspace, void *stream, unsigned pack_size) {
    reset_nv_gpu<Ldata, LIdata>(workspace, desc->y_data_size, pack_size, desc->max_grid_size, stream);
    sum_nv_gpu<Tdata, TIdata, Ldata, LIdata>(workspace, x, desc->data_size, desc->x_per_NC_data_size, pack_size, desc->max_grid_size, stream);
    average_nv_gpu<Ldata, LIdata, Tdata, TIdata>(y, workspace, desc->y_data_size, desc->x_per_NC_data_size, pack_size, desc->max_grid_size, stream);
}

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata>
void launch_global_avg_pool_folding(GlobalAvgPoolCudaDescriptor_t desc, void *y, void const *x, void *workspace, uint64_t workspace_size, void *stream, unsigned pack_size) {
    if (workspace_size == 0) {
        global_avg_pool_folding_direct<Tdata, TIdata, 256>(desc, y, x, stream, pack_size);
    } else {
        global_avg_pool_folding_workspace<Tdata, TIdata, Ldata, LIdata, 256>(desc, y, x, workspace, stream, pack_size);
    }
}

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata>
void global_avg_pool_nv_gpu_hd(GlobalAvgPoolCudaDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream, unsigned pack_size) {
    if (desc->data_size == 0) {
        return;
    }
    if (desc->x_per_NC_data_size <= desc->max_block_size) {
        const auto y_ = reinterpret_cast<Tdata *>(y);
        const auto x_ = reinterpret_cast<Tdata const *>(x);
        launch_global_avg_pool_padding<Tdata, TIdata, Ldata, LIdata>(desc, y_, x_, stream, pack_size);
    } else {
        launch_global_avg_pool_folding<Tdata, TIdata, Ldata, LIdata>(desc, y, x, workspace, workspace_size, stream, pack_size);
    }
}

template<typename Tdata, typename TIdata, typename Ldata, typename LIdata>
infiniopStatus_t global_avg_pool_nv_gpu(GlobalAvgPoolCudaDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream, unsigned pack_size) {
    // use cuDNN lib
    if (desc->ndim <= 4) {
        checkCudnnError(use_cudnn(desc->cudnn_handles_t, desc->device_id,
                                  [&](cudnnHandle_t handle) { return cudnnPoolingForward(handle, desc->pool_desc,
                                                                                         &desc->alpha, desc->x_desc, x, &desc->beta,
                                                                                         desc->y_desc, y); }));
    } else {
        global_avg_pool_nv_gpu_hd<Tdata, TIdata, Ldata, LIdata>(desc, workspace, workspace_size, y, x, stream, pack_size);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaGlobalAvgPool(GlobalAvgPoolCudaDescriptor_t desc,
                                   void *workspace, uint64_t workspace_size,
                                   void *y, void const *x,
                                   void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return global_avg_pool_nv_gpu<half, half, float, float>(desc, workspace, workspace_size, y, x, stream, 1);
    }
    if (desc->dtype == F32) {
        return global_avg_pool_nv_gpu<float, float, float, float>(desc, workspace, workspace_size, y, x, stream, 1);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
