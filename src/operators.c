#include "internal.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef ENABLE_CPU
#include "cpu/cpu.h"
#endif

#ifdef ENABLE_NV_GPU
#include "nv_gpu/nv_gpu.cuh"
#endif

Op op_create(Device dev, Optype opty, void *config) {
    char *err = NULL;
    int err_len = SIZE_MAX;
    switch (dev) {
#ifdef ENABLE_CPU
        case DevCpu:
            return op_create_cpu(opty, config);
#endif
#ifdef ENABLE_NV_GPU
        case DevNvGpu:
            return op_create_nv_gpu(opty, config);
#endif
        default:
            PANIC(UnsupportedDevice);
            return NULL;
    }
}
void op_destroy(Op op) {
    op->drop(op);
}

Kn kn_load(Op op, void *rt_ctx) {
    return op->load(op, rt_ctx);
}
void kn_unload(Kn kn) {
    kn->drop(kn);
}

void *fn_get(Kn kn) {
    return kn->fn;
}
