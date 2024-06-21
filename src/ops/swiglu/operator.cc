#include "../utils.h"
#include "swiglu.h"

#ifdef ENABLE_CPU
#include "cpu/swiglu_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/swiglu.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/swiglu_cnnl.h"
#endif

struct SwigluDescriptor {
    Device device;
};

__C void *createSwigluDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return (SwigluDescriptor *) (new SwigluCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return (SwigluDescriptor *) (new SwigluCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu:
        return (SwigluDescriptor *) (new SwigluBangDescriptor(device));
#endif
    default:
        PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroySwigluDescriptor(void *descriptor) {
    delete (SwigluDescriptor *) descriptor;
}

__C void swiglu(void *descriptor, MutTensor gate, ConstTensor up, void *stream) {
    auto desc = reinterpret_cast<SwigluDescriptor *>(descriptor);
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            swiglu_cpu_f16(gate, up);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            swiglu_nv_gpu_f16(gate, up, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            swiglu_cnnl_f16(gate, up, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
