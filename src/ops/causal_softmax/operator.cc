#include "../utils.h"
#include "causal_softmax.h"
#include "causal_softmax_config.h"

#ifdef ENABLE_CPU
#include "cpu/causal_softmax_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "../../devices/cuda/common_cuda.h"
#include "cuda/causal_softmax.cuh"
#endif

struct CausalSoftmaxDescriptor {
    Device device;
};

__C CausalSoftmaxDescriptor *createCausalSoftmaxDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (CausalSoftmaxDescriptor *) (new CausalSoftmaxCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu: {
            ASSERT_VALID_PTR(config);
            CausalSoftmaxCudaConfig *cuda_config = (CausalSoftmaxCudaConfig *) config;
            return (CausalSoftmaxDescriptor *) (new CausalSoftmaxCudaDescriptor{
                device,
                ROUND_UP_DIV(cuda_config->max_dim, MAX_THREADS_PER_BLOCK)});
        }

#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
}

__C void destroyCausalSoftmaxDescriptor(CausalSoftmaxDescriptor *descriptor) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            delete (CausalSoftmaxCpuDescriptor *) (descriptor);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            delete (CausalSoftmaxCudaDescriptor *) (descriptor);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}

__C void causalSoftmax(CausalSoftmaxDescriptor *descriptor, Tensor y, void *stream) {
    switch (descriptor->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            causal_softmax_cpu_f16(y);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            causal_softmax_nv_gpu_f16((CausalSoftmaxCudaDescriptor *) descriptor, y, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
