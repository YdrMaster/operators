#ifndef __CPU_ADD_H__
#define __CPU_ADD_H__

#include "operators.h"
#include <numeric>
struct AddCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t data_size;
};

typedef struct AddCpuDescriptor *AddCpuDescriptor_t;

infiniopStatus_t cpuCreateAddDescriptor(infiniopHandle_t,
                                        AddCpuDescriptor_t *,
                                        infiniopTensorDescriptor_t c,
                                        infiniopTensorDescriptor_t a,
                                        infiniopTensorDescriptor_t b);

infiniopStatus_t cpuAdd(AddCpuDescriptor_t desc,
                        void *workspace,
                        uint64_t workspace_size,
                        void *c, void *a, void *b,
                        void *stream);

infiniopStatus_t cpuDestroyAddDescriptor(AddCpuDescriptor_t desc);

#endif
