#ifndef __CPU_GLOBAL_AVG_POOL_H__
#define __CPU_GLOBAL_AVG_POOL_H__

#include "operators.h"
#include <numeric>

struct GlobalAvgPoolCpuDescriptor {
    Device device;
    DT dtype;
    uint64_t y_data_size;
    uint64_t x_per_NC_data_size;
};

typedef struct GlobalAvgPoolCpuDescriptor *GlobalAvgPoolCpuDescriptor_t;

infiniopStatus_t cpuCreateGlobalAvgPoolDescriptor(infiniopHandle_t,
                                                  GlobalAvgPoolCpuDescriptor_t *,
                                                  infiniopTensorDescriptor_t y,
                                                  infiniopTensorDescriptor_t x);

infiniopStatus_t cpuGetGlobalAvgPoolWorkspaceSize(GlobalAvgPoolCpuDescriptor_t desc, uint64_t *size);

infiniopStatus_t cpuGlobalAvgPool(GlobalAvgPoolCpuDescriptor_t desc,
                                  void *workspace, uint64_t workspace_size, void *y, void const *x,
                                  void *stream);

infiniopStatus_t cpuDestroyGlobalAvgPoolDescriptor(GlobalAvgPoolCpuDescriptor_t desc);

#endif
