#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "add.cuh"

struct half4 {
    __half x, y, z, w;

    __device__ half4 operator+(const half4 &other) const {
        return half4{__hadd(x, other.x), __hadd(y, other.y), __hadd(z, other.z), __hadd(w, other.w)};
    }
};

__device__ uint64_t getDstIndex(uint64_t flat_index, uint64_t ndim, int64_t const *src_strides, int64_t const *dst_strides) {
    uint64_t res = 0;
    for (uint64_t i = 0; i < ndim; ++i) {
        res += flat_index / src_strides[i] * dst_strides[i];
        flat_index %= src_strides[i];
    }
    return res;
}

template<typename Tdata, typename BTdata>
__global__ void add(
    Tdata *c,
    const Tdata *a,
    const Tdata *b,
    const int64_t *a_strides,
    const int64_t *b_strides,
    const int64_t *c_strides,
    uint64_t data_size,
    uint64_t ndim,
    uint64_t offset,
    bool broadcasted,
    unsigned pack_size) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < data_size) {
        if (broadcasted) {
            idx *= pack_size;
            auto a_ = reinterpret_cast<const BTdata *>(a);
            auto b_ = reinterpret_cast<const BTdata *>(b);
            auto c_ = reinterpret_cast<BTdata *>(c);
            for (size_t i = 0; i < pack_size; ++i) {
                auto a_idx = getDstIndex(idx + i, ndim, c_strides, a_strides);
                auto b_idx = getDstIndex(idx + i, ndim, c_strides, b_strides);
                c_[idx + i] = a_[a_idx] + b_[b_idx];
            }
            return;
        }
        c[idx] = a[idx] + b[idx];
    }
}

void add_nv_gpu_f16(AddCudaDescriptor_t desc, void *c, void const *a, void const *b, void *stream) {
    auto data_size = desc->c_data_size / 4;
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(MAX_THREADS_PER_BLOCK), data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    auto a_ptr = reinterpret_cast<const half4 *>(a);
    auto b_ptr = reinterpret_cast<const half4 *>(b);
    auto c_ptr = reinterpret_cast<half4 *>(c);

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    for (uint64_t i = 0; i < data_size; i += step) {
        add<half4, half><<<gridDims, blockDims, 0, cuda_stream>>>(
            c_ptr, a_ptr, b_ptr, desc->a_strides, desc->b_strides, desc->c_strides, data_size, desc->ndim, i, desc->broadcasted, 4);
    }
}

infiniopStatus_t cudaAdd(AddCudaDescriptor_t desc,
                         void *c, void const *a, void const *b,
                         void *stream) {
    if (!dtype_eq(desc->dtype, F16)) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    checkCudaError(cudaSetDevice(desc->device_id));
    add_nv_gpu_f16(desc, c, a, b, stream);
    return STATUS_SUCCESS;
}
