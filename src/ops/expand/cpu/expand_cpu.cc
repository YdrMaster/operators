#include "expand_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateExpandDescriptor(infiniopHandle_t,
                                           ExpandCpuDescriptor_t *desc_ptr,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x) {
    uint64_t ndim = y->ndim;
    if (!isValidBroadcastShape(y, x)) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    if (y->dt != x->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t y_data_size = std::accumulate(y->shape, y->shape + y->ndim, 1ULL, std::multiplies<uint64_t>());

    // get the adjusted strides for x in terms of y
    int64_t *x_strides = new int64_t[ndim];
#pragma omp parallel for
    for (size_t i = 0; i < ndim; ++i) {
        x_strides[i] = (i < ndim - x->ndim || y->shape[i] != x->shape[i + x->ndim - ndim]) ? 0 : x->strides[i + x->ndim - ndim];
    }

    *desc_ptr = new ExpandCpuDescriptor{
        DevCpu,
        y->dt,
        ndim,
        y_data_size,
        x_strides,
        y->strides,
    };

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyExpandDescriptor(ExpandCpuDescriptor_t desc) {
    delete[] desc->x_strides;
    delete desc;
    return STATUS_SUCCESS;
}

template<typename Tdata>
infiniopStatus_t expand_cpu(ExpandCpuDescriptor_t desc, void *y, void const *x) {
    auto x_ = reinterpret_cast<Tdata const *>(x);
    auto y_ = reinterpret_cast<Tdata *>(y);

#pragma omp parallel for
    for (uint64_t i = 0; i < desc->y_data_size; ++i) {
        y_[i] = x_[getDstOffset(i, desc->ndim, desc->y_strides, desc->x_strides)];
    }
    return STATUS_SUCCESS;
}

infiniopStatus_t cpuExpand(ExpandCpuDescriptor_t desc,
                           void *y, void const *x,
                           void *stream) {
    if (desc->dtype == F16) {
        return expand_cpu<uint16_t>(desc, y, x);
    }
    if (desc->dtype == F32) {
        return expand_cpu<float>(desc, y, x);
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
