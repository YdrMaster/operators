#include "../utils.h"
#include "operators.h"
#include "ops/global_avg_pool/global_avg_pool.h"

#ifdef ENABLE_CPU
#include "cpu/global_avg_pool_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/global_avg_pool.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
// TODO: Cambricon
#endif

__C infiniopStatus_t infiniopCreateGlobalAvgPoolDescriptor(
    infiniopHandle_t handle,
    infiniopGlobalAvgPoolDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateGlobalAvgPoolDescriptor(handle, (GlobalAvgPoolCpuDescriptor_t *) desc_ptr, y, x);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateGlobalAvgPoolDescriptor((CudaHandle_t) handle, (GlobalAvgPoolCudaDescriptor_t *) desc_ptr, y, x);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopGetGlobalAvgPoolWorkspaceSize(infiniopGlobalAvgPoolDescriptor_t desc, uint64_t *size) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGetGlobalAvgPoolWorkspaceSize((GlobalAvgPoolCpuDescriptor_t) desc, size);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGetGlobalAvgPoolWorkspaceSize((GlobalAvgPoolCudaDescriptor_t) desc, size);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO: Cambricon support
#endif
    }
    return STATUS_BAD_DEVICE;
}


__C infiniopStatus_t infiniopGlobalAvgPool(infiniopGlobalAvgPoolDescriptor_t desc, void *workspace, uint64_t workspace_size, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuGlobalAvgPool((GlobalAvgPoolCpuDescriptor_t) desc, workspace, workspace_size, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaGlobalAvgPool((GlobalAvgPoolCudaDescriptor_t) desc, workspace, workspace_size, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyGlobalAvgPoolDescriptor(infiniopGlobalAvgPoolDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyGlobalAvgPoolDescriptor((GlobalAvgPoolCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyGlobalAvgPoolDescriptor((GlobalAvgPoolCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
