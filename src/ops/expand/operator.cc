#include "../utils.h"
#include "operators.h"
#include "ops/expand/expand.h"

#ifdef ENABLE_CPU
#include "cpu/expand_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/cuda_handle.h"
#include "cuda/expand.cuh"
#endif

__C infiniopStatus_t infiniopCreateExpandDescriptor(
    infiniopHandle_t handle,
    infiniopExpandDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x) {
    switch (handle->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuCreateExpandDescriptor(handle, (ExpandCpuDescriptor_t *) desc_ptr, y, x);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaCreateExpandDescriptor((CudaHandle_t) handle, (ExpandCudaDescriptor_t *) desc_ptr, y, x);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopExpand(infiniopExpandDescriptor_t desc, void *y, void const *x, void *stream) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuExpand((ExpandCpuDescriptor_t) desc, y, x, stream);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaExpand((ExpandCudaDescriptor_t) desc, y, x, stream);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}

__C infiniopStatus_t infiniopDestroyExpandDescriptor(infiniopExpandDescriptor_t desc) {
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return cpuDestroyExpandDescriptor((ExpandCpuDescriptor_t) desc);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return cudaDestroyExpandDescriptor((ExpandCudaDescriptor_t) desc);
        }

#endif
#ifdef ENABLE_CAMBRICON_MLU
        // TODO
#endif
    }
    return STATUS_BAD_DEVICE;
}
