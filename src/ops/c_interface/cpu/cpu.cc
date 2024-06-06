#include "cpu.h"
#include "../../rms_norm/cpu/rms_norm.h"
#include "../../rotary_embedding/cpu/rotary_embedding.h"

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
                (Fn) rms_norm_cpu_f16,
                kn_drop,
            };
            return kn;
        }
        case OpMatMul:
            return nullptr;
        case OpRotaryEmbedding: {
            auto kn = new Kernel{
                DevCpu,
                OpRotaryEmbedding,
                nullptr,
                (Fn) rotary_embedding_cpu_f16,
                kn_drop,
            };
            return kn;
        }
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
        case OpRotaryEmbedding: {
            auto op = new Operator{
                DevCpu,
                OpRotaryEmbedding,
                nullptr,
                load,
                op_drop,
            };
            return op;
        }
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
