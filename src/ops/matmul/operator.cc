#include "matmul.h"

#ifdef ENABLE_CPU
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_NV_GPU
#include "cuda/matmul_cuda.h"
#include <cublas_v2.h>
#endif

#include "../utils.h"

extern "C" void *createMatmulDescriptor(Device device, void *config) {
    auto desc = new MatmulDescriptor{device};
    return (void *) desc;
}

extern "C" void destroyMatmulDescriptor(void *descriptor) {
    auto desc = (MatmulDescriptor *) descriptor;
    delete desc;
}

extern "C" void matmul(void *descriptor, MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha, void *stream) {
    auto desc = (MatmulDescriptor *) descriptor;
    switch (desc->device) {
#ifdef ENABLE_CPU
        case DevCpu:
            matmul_cpu_f16(c, beta, a, b, alpha);
            break;
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            matmul_nv_gpu_f16(c, beta, a, b, alpha, stream);
            break;
#endif
        default:
            PANIC(UnsupportedDevice);
    }
}
