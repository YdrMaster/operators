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


__C void *createSwigluDescriptor(Device device, void *config) {
    SwigluDescriptor *desc = new SwigluDescriptor{device};
    return (void *) desc;
};

__C void destroySwigluDescriptor(void *descriptor) {
    SwigluDescriptor *desc = (SwigluDescriptor *) descriptor;
    delete desc;
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
