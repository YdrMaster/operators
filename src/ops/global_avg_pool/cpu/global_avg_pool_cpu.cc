#include "global_avg_pool_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateGlobalAvgPoolDescriptor(infiniopHandle_t,
                                                  GlobalAvgPoolCpuDescriptor_t *desc_ptr,
                                                  infiniopTensorDescriptor_t y,
                                                  infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (ndim < 2 || ndim != x->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (i < 2 && y->shape[i] != x->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        } else if (i >= 2 && y->shape[i] != 1) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!is_contiguous(y) || !is_contiguous(x)) {
        return STATUS_BAD_TENSOR_STRIDES;
    }
    if (y->dt != F16 && y->dt != F32) {
        return STATUS_BAD_TENSOR_DTYPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + 2, 1ULL, std::multiplies<uint64_t>());
    uint64_t x_per_NC_data_size = std::accumulate(x->shape + 2, x->shape + ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new GlobalAvgPoolCpuDescriptor{
        DevCpu,
        y->dt,
        y_data_size,
        x_per_NC_data_size,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolCpuDescriptor_t desc, uint64_t *size) {
    *size = 0;
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t global_avg_pool_cpu(GlobalAvgPoolCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);
    const auto x_size = desc->x_per_NC_data_size;

    for (uint64_t i = 0; i < desc->y_data_size; ++i) {
        if constexpr (std::is_same<Tdata, uint16_t>::value) {
            float sum = std::accumulate(x_ + i * x_size, x_ + (i + 1) * x_size, 0.0f,
                                        [](float res, uint16_t value) {
                                            return res + f16_to_f32(value);
                                        });
            y_[i] = f32_to_f16(sum / x_size);
        } else {
            y_[i] = std::accumulate(x_ + i * x_size, x_ + (i + 1) * x_size, 0) / x_size;
        }
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuGlobalAvgPool(GlobalAvgPoolCpuDescriptor_t desc,
                                  void *workspace, uint64_t workspace_size, void *y, void const *x,
                                  void *stream) {
    if (desc->dtype == F16) {
        return global_avg_pool_cpu<uint16_t>(desc, y, x);
    }
    if (desc->dtype == F32) {
        return global_avg_pool_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
