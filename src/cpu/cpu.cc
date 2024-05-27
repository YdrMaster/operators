#include "cpu.h"
#include "rms_norm.h"
#include <cstdio>

void kn_drop(Kn kn) {
    delete kn;
}

Kn load(Op op, void *rt_ctx) {
    switch (op->optype) {
        case OpRmsNorm: {
            auto kn = new Kernel{
                DevCpu,
                OpRmsNorm,
                rms_norm_cpu_f16,
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

void op_drop(Op op) {
    delete op;
}

Op op_create_cpu(Optype opty, void *config) {
    switch (opty) {
        case OpRmsNorm: {
            auto op = new Operator{
                DevCpu,
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
