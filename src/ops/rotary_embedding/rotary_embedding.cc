#include "rotary_embedding.h"
#ifdef ENABLE_CPU
#include "cpu/rotary_embedding_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/rotary_embedding.cuh"
#endif

#include "../utils.h"

extern "C" void *createRotaryEmbeddingDescriptor(Device device, void *config) {
    RotaryEmbeddingDescriptor *desc = new RotaryEmbeddingDescriptor{device};
    return (void *) desc;
};

extern "C" void destroyRotaryEmbeddingDescriptor(void *descriptor) {
    RotaryEmbeddingDescriptor *desc = (RotaryEmbeddingDescriptor *) descriptor;
    delete desc;
}

extern "C" void rotaryEmbedding(void *descriptor, MutTensor t, ConstTensor pos, float theta, void *stream) {
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
        default:
            PANIC(UnsupportedDevice);
    }
};
