#ifndef __CPU_EXPAND_H__
#define __CPU_EXPAND_H__

#include "operators.h"
#include <cstring>
#include <numeric>

struct ExpandCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t ndim;
    uint64_t y_data_size;
    int64_t const *x_strides;
    int64_t const *y_strides;
};

typedef struct ExpandCpuDescriptor *ExpandCpuDescriptor_t;

infiniopStatus_t cpuCreateExpandDescriptor(infiniopHandle_t,
                                           ExpandCpuDescriptor_t *,
                                           infiniopTensorDescriptor_t y,
                                           infiniopTensorDescriptor_t x);

infiniopStatus_t cpuExpand(ExpandCpuDescriptor_t desc,
                           void *y, void const *x, void *stream);

infiniopStatus_t cpuDestroyExpandDescriptor(ExpandCpuDescriptor_t desc);

#endif
