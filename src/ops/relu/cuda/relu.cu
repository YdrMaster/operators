#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "relu.cuh"

/**
 * @brief A templated vector struct that supports applying relu on arrays.
 *
 * @tparam T - The access data type for elements in the vector.
 * @tparam TComp - The computation data type used for arithmetic operations. 
 * @tparam N - The number of elements of type T in the vector for a single access.
 */
template<typename T, typename TComp, size_t N>
struct vecN {
    T data[N];
    constexpr static size_t pack_size = sizeof(T) / sizeof(TComp);

    __device__ __forceinline__ constexpr vecN(const TComp &val) {
        const auto data_ = reinterpret_cast<TComp *>(data);
        const auto size = N * pack_size;
#pragma unroll
        for (size_t i = 0; i < size; ++i) {
            data_[i] = 0;
        }
    }

    __device__ __forceinline__ vecN<T, TComp, N> &operator=(const vecN<T, TComp, N> &other) {
        if constexpr (std::is_same<T, TComp>::value) {
#pragma unroll
            for (int i = 0; i < N; ++i) {
                data[i] = other.data[i] < TComp(0) ? TComp(0) : other.data[i];
            }
        } else {
            auto *data_this = reinterpret_cast<vecN<TComp, TComp, pack_size> *>(data);
            auto *data_other = reinterpret_cast<const vecN<TComp, TComp, pack_size> *>(other.data);
#pragma unroll
            for (int i = 0; i < N; ++i) {
                data_this[i] = data_other[i];
            }
        }
        return *this;
    }

    __device__ __forceinline__ bool operator<(const vecN<T, TComp, N> &other) const {
        return false;
    }

    __device__ __forceinline__ const T &operator[](size_t i) const {
        return data[i];
    }
};

template<typename Tdata>
__global__ void relu(
    Tdata *y,
    const Tdata *x,
    uint64_t data_size,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        y[idx] = x[idx] < Tdata(0) ? Tdata(0) : x[idx];
    }
}

template<typename Tdata>
void relu_nv_gpu(ReluCudaDescriptor_t desc, Tdata *y, Tdata const *x, uint64_t data_size, uint64_t offset, void *stream) {
    if (data_size == 0) {
        return;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < data_size; i += step) {
        relu<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(y, x, offset + data_size, offset + i);
    }
}

template<typename Tdata, typename TIdata>
infiniopStatus_t relu_nv_gpu(ReluCudaDescriptor_t desc, void *y, void const *x, void *stream, uint64_t pack_size) {
    const auto data_size = desc->data_size / pack_size;
    const auto x_vec = reinterpret_cast<const Tdata *>(x);
    const auto y_vec = reinterpret_cast<Tdata *>(y);
    relu_nv_gpu(desc, y_vec, x_vec, data_size, 0, stream);

    const auto remainder = desc->data_size % pack_size;
    const auto x_ = reinterpret_cast<const TIdata *>(x);
    const auto y_ = reinterpret_cast<TIdata *>(y);
    relu_nv_gpu(desc, y_, x_, remainder, data_size * pack_size, stream);
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaRelu(ReluCudaDescriptor_t desc,
                          void *y, void const *x,
                          void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return relu_nv_gpu<vecN<half, half, 4>, half>(desc, y, x, stream, 4);
    }
    if (desc->dtype == F32) {
        return relu_nv_gpu<vecN<float2, float, 2>, float>(desc, y, x, stream, 4);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
