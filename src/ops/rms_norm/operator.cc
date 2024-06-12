#include "../utils.h"
#include "rms_norm.h"

#ifdef ENABLE_CPU
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rms_norm.cuh"
#endif

__C void *createRMSNormDescriptor(Device device, void *config) {
    return new RMSNormDescriptor{device};
}

__C void destroyRMSNormDescriptor(void *descriptor) {
    delete (RMSNormDescriptor *) descriptor;
}

__C void rmsNorm(void *descriptor, MutTensor y, ConstTensor x, ConstTensor w, float epsilon, void *stream) {
    auto desc = (RMSNormDescriptor *) descriptor;
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            rms_norm_cpu_f16(y, x, w, epsilon);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            rms_norm_nv_gpu_f16(y, x, w, epsilon, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
