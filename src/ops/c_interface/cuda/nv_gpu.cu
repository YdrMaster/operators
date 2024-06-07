#include "../../../utils.h"
#include "../../reform/cuda/reform.cuh"
#include "../../rms_norm/cuda/rms_norm.cuh"
#include "../../rotary_embedding/cuda/rotary_embedding.cuh"
#include "nv_gpu.cuh"
#include <cstdio>

static void kn_drop(Kn kn) {
    delete reinterpret_cast<NvGpuRtCtx *>(kn->rt_ctx);
    delete kn;
}

#define KN_LOAD_GPU(op, ctx, fn) \
    case op: {                   \
        auto kn = new Kernel{    \
            DevNvGpu,            \
            op,                  \
            ctx,                 \
            (Fn) fn,             \
            kn_drop,             \
        };                       \
        return kn;               \
    }

static Kn load(Op op, void *rt_ctx) {
    ASSERT_VALID_PTR(rt_ctx);
    auto ctx = new NvGpuRtCtx(*reinterpret_cast<NvGpuRtCtx *>(rt_ctx));
    switch (op->optype) {
        KN_LOAD_GPU(OpRmsNorm, ctx, rms_norm_nv_gpu_f16)
        KN_LOAD_GPU(OpRotaryEmbedding, ctx, rotary_embedding_nv_gpu_f16)
        KN_LOAD_GPU(OpReform, ctx, reform_nv_gpu)
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
#undef KN_LOAD_GPU

static void op_drop(Op op) {
    delete op;
}

#define OP_CREATE_GPU(optype, config) \
    case optype: {                    \
        auto op = new Operator{       \
            DevNvGpu,                 \
            optype,                   \
            config,                   \
            load,                     \
            op_drop,                  \
        };                            \
        return op;                    \
    }

Op op_create_nv_gpu(Optype opty, void *) {
    switch (opty) {
        OP_CREATE_GPU(OpRmsNorm, nullptr)
        OP_CREATE_GPU(OpRotaryEmbedding, nullptr)
        OP_CREATE_GPU(OpReform, nullptr)
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
#undef OP_CREATE_GPU
