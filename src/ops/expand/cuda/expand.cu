#include "../../../devices/cuda/common_cuda.h"
#include "../../utils.h"
#include "expand.cuh"

template<typename Tdata>
__global__ void expand(
    Tdata *y,
    const Tdata *x,
    const int64_t *y_strides,
    const int64_t *x_strides,
    const uint64_t *y_shape,
    uint64_t y_data_size,
    uint64_t ndim,
    uint64_t offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (idx < y_data_size) {
        uint64_t y_idx = getOffset(idx, ndim, y_shape, y_strides);
        y[y_idx] = x[getDstOffset(y_idx, ndim, y_strides, x_strides)];
    }
}

template<typename Tdata>
infiniopStatus_t expand_nv_gpu(ExpandCudaDescriptor_t desc, void *y, void const *x, void *stream) {
    if (desc->y_data_size == 0) {
        return STATUS_SUCCESS;
    }
    dim3 blockDims = dim3(std::min(static_cast<uint64_t>(256), desc->y_data_size));
    dim3 gridDims = dim3(std::min(ROUND_UP_DIV(desc->y_data_size, blockDims.x), desc->max_grid_size));
    uint64_t step = gridDims.x * blockDims.x;

    const auto x_ = reinterpret_cast<Tdata const *>(x);
    const auto y_ = reinterpret_cast<Tdata *>(y);
    const auto x_strides = reinterpret_cast<int64_t const *>(desc->strides_and_shape_d);
    const auto y_strides = reinterpret_cast<int64_t const *>(desc->strides_and_shape_d + desc->ndim * sizeof(int64_t));
    const auto y_shape = reinterpret_cast<uint64_t const *>(desc->strides_and_shape_d + 2 * desc->ndim * sizeof(int64_t));
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

#pragma unroll
    for (uint64_t i = 0; i < desc->y_data_size; i += step) {
        expand<Tdata><<<gridDims, blockDims, 0, cuda_stream>>>(
            y_, x_, y_strides, x_strides, y_shape, i + desc->y_data_size, desc->ndim, i);
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cudaExpand(ExpandCudaDescriptor_t desc,
                            void *y, void const *x,
                            void *stream) {
    checkCudaError(cudaSetDevice(desc->device_id));
    if (desc->dtype == F16) {
        return expand_nv_gpu<half>(desc, y, x, stream);
    }
    if (desc->dtype == F32) {
        return expand_nv_gpu<float>(desc, y, x, stream);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
