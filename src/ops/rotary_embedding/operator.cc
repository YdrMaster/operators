#include "../utils.h"
#include "rotary_embedding.h"

#ifdef ENABLE_CPU
#include "cpu/rotary_embedding_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rotary_embedding.cuh"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "bang/rotary_embedding_cnnl.h"
#endif

struct RotaryEmbeddingDescriptor {
    Device device;
};

__C void *createRotaryEmbeddingDescriptor(Device device, void *config) {
    switch (device) {
#ifdef ENABLE_CPU
        case DevCpu:
            return (RotaryEmbeddingDescriptor *) (new RotaryEmbeddingCpuDescriptor{device});
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return (RotaryEmbeddingDescriptor *) (new RotaryEmbeddingCudaDescriptor{device});
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            return (RotaryEmbeddingDescriptor *) (new RotaryEmbeddingBangDescriptor(device));
#endif
        default:
            PANIC(UnsupportedDevice);
    }
    return nullptr;
};

__C void destroyRotaryEmbeddingDescriptor(void *descriptor) {
    delete (RotaryEmbeddingDescriptor *) descriptor;
}

__C void rotaryEmbedding(void *descriptor, MutTensor t, ConstTensor pos, float theta, void *stream) {
    auto desc = reinterpret_cast<RotaryEmbeddingDescriptor *>(descriptor);
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            rotary_embedding_cpu_f16(t, pos, theta);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            rotary_embedding_nv_gpu_f16(t, pos, theta, stream);
            break;
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu:
            rotary_embedding_cnnl_f16(t, pos, theta, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
};
