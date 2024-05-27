#include "nv_gpu.cuh"
#include <cstdio>

static void kn_drop(Kn kn) {
    delete kn;
}

static Kn load(Op op, void *rt_ctx) {
    switch (op->optype) {
        case OpRmsNorm: {
            auto kn = new Kernel{
                DevNvGpu,
                OpRmsNorm,
                nullptr,
                kn_drop,
            };
            return kn;
        }
        case OpMatMul:
            return nullptr;
        case OpRotaryEmbedding:
            return nullptr;
        case OpReform:
            return nullptr;
        case OpCausalSoftmax:
            return nullptr;
        case OpSwiglu:
            return nullptr;
        default:
            return nullptr;
    }
}

static void op_drop(Op op) {
    delete op;
}

Op op_create_nv_gpu(Optype opty, void *config) {
    switch (opty) {
        case OpRmsNorm: {
            auto op = new Operator{
                DevNvGpu,
                OpRmsNorm,
                load,
                op_drop,
            };
            return op;
        }
        case OpMatMul:
            return nullptr;
        case OpRotaryEmbedding:
            return nullptr;
        case OpReform:
            return nullptr;
        case OpCausalSoftmax:
            return nullptr;
        case OpSwiglu:
            return nullptr;
        default:
            return nullptr;
    }
}
