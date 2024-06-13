#include "../utils.h"
#include "reform.h"

#ifdef ENABLE_CPU
#include "cpu/reform_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/reform.cuh"
#endif

struct ReformDescriptor {
    Device device;
};

__C ReformDescriptor *createReformDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (ReformDescriptor *) (new ReformCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            return (ReformDescriptor *) (new ReformCudaDescriptor{device});
        }

#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyReformDescriptor(ReformDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (ReformCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (ReformCudaDescriptor *) (descriptor);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void reform(void *descriptor, MutTensor y, ConstTensor x, void *stream) {
    auto desc = reinterpret_cast<ReformDescriptor *>(descriptor);
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            reform_cpu(y, x);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            reform_nv_gpu(y, x, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
