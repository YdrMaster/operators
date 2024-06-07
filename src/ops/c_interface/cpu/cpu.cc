#include "cpu.h"
#include "../../reform/cpu/reform.h"
#include "../../rms_norm/cpu/rms_norm.h"
#include "../../rotary_embedding/cpu/rotary_embedding.h"

static void kn_drop(Kn kn) {
    delete kn;
}

#define KN_LOAD_CPU(op, rt_ctx, fn) \
    case op: {                      \
        auto kn = new Kernel{       \
            DevCpu,                 \
            op,                     \
            rt_ctx,                 \
            (Fn) fn,                \
            kn_drop,                \
        };                          \
        return kn;                  \
    }

static Kn load(Op op, void *) {
    switch (op->optype) {
        KN_LOAD_CPU(OpRmsNorm, nullptr, rms_norm_cpu_f16)
        KN_LOAD_CPU(OpRotaryEmbedding, nullptr, rotary_embedding_cpu_f16)
        KN_LOAD_CPU(OpReform, nullptr, reform_cpu)
        case OpMatMul:
            return nullptr;
        case OpCausalSoftmax:
            return nullptr;
        case OpSwiglu:
            return nullptr;
        default:
            return nullptr;
    }
}
#undef KN_LOAD_CPU

static void op_drop(Op op) {
    delete op;
}

#define OP_LOAD_CPU(optype, config) \
    case optype: {                  \
        auto op = new Operator{     \
            DevCpu,                 \
            optype,                 \
            config,                 \
            load,                   \
            op_drop,                \
        };                          \
        return op;                  \
    }

Op op_create_cpu(Optype opty, void *) {
    switch (opty) {
        OP_LOAD_CPU(OpRmsNorm, nullptr)
        OP_LOAD_CPU(OpRotaryEmbedding, nullptr)
        OP_LOAD_CPU(OpReform, nullptr)
        case OpMatMul:
            return nullptr;
        case OpCausalSoftmax:
            return nullptr;
        case OpSwiglu:
            return nullptr;
        default:
            return nullptr;
    }
}
#undef OP_LOAD_CPU
