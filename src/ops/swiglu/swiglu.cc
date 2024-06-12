#include "swiglu.h"
#ifdef ENABLE_CPU
#include "cpu/swiglu_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/swiglu.cuh"
#endif

#include "../utils.h"

extern "C" void *createSwigluDescriptor(Device device, void *config) {
    SwigluDescriptor *desc = new SwigluDescriptor{device};
    return (void *) desc;
};

extern "C" void destroySwigluDescriptor(void *descriptor) {
    SwigluDescriptor *desc = (SwigluDescriptor *) descriptor;
    delete desc;
}

extern "C" void swiglu(void *descriptor, MutTensor gate, ConstTensor up, void *stream) {
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
        default:
            PANIC(UnsupportedDevice);
    }
};
