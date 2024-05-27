#include "cpu.h"
#include "rms_norm.h"
#include <cstdio>

static void kn_drop(Kn kn) {
    delete kn;
}

static Kn load(Op op, void *) {
    switch (op->optype) {
        case OpRmsNorm: {
            auto kn = new Kernel{
                DevCpu,
                OpRmsNorm,
                nullptr,
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

static void op_drop(Op op) {
    delete op;
}

Op op_create_cpu(Optype opty, void *) {
    switch (opty) {
        case OpRmsNorm: {
            auto op = new Operator{
                DevCpu,
                OpRmsNorm,
                nullptr,
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
