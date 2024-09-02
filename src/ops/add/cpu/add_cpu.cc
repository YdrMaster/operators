#include "add_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"

infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *desc_ptr,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b) {
    uint64_t ndim = c->ndim;
    if (ndim != a->ndim || ndim != b->ndim) {
        return STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < ndim; ++i) {
        if (a->shape[i] != b->shape[i] || a->shape[i] != c->shape[i]) {
            return STATUS_BAD_TENSOR_SHAPE;
        }
    }
    if (!dtype_eq(c->dt, F16) || c->dt != a->dt || c->dt != b->dt) {
        return STATUS_BAD_TENSOR_DTYPE;
    }

    uint64_t data_size = std::accumulate(a->shape, a->shape + ndim, 1ULL, std::multiplies<uint64_t>());

    *desc_ptr = new AddCpuDescriptor{
        DevCpu,
        c->dt,
        data_size};

    return STATUS_SUCCESS;
}

infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc) {
    delete desc;
    return STATUS_SUCCESS;
}

void add_cpu_f16(AddCpuDescriptor_t desc, void *c, void const *a, void const *b) {
    auto a_ = reinterpret_cast<uint16_t const *>(a);
    auto b_ = reinterpret_cast<uint16_t const *>(b);
    auto c_ = reinterpret_cast<uint16_t *>(c);
    for (uint64_t i = 0; i < desc->data_size; ++i) {
        c_[i] = f32_to_f16(f16_to_f32(a_[i]) + f16_to_f32(b_[i]));
    }
}

infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *c, void const *a, void const *b,
                        void *stream) {
    if (dtype_eq(desc->dtype, F16)) {
        add_cpu_f16(desc, c, a, b);
        return STATUS_SUCCESS;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
