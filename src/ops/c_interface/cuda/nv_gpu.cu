#include "nv_gpu.cuh"
#include "../../rms_norm/cuda/rms_norm.cuh"
#include "../../../utils.h"
#include <cstdio>

static void kn_drop(Kn kn) {
    delete reinterpret_cast<NvGpuRtCtx *>(kn->rt_ctx);
    delete kn;
}

static Kn load(Op op, void *rt_ctx) {
    ASSERT_VALID_PTR(rt_ctx);
    auto ctx = new NvGpuRtCtx(*reinterpret_cast<NvGpuRtCtx *>(rt_ctx));
    switch (op->optype) {
        case OpRmsNorm: {
            auto kn = new Kernel{
                DevNvGpu,
                OpRmsNorm,
                ctx,
               (Fn) rms_norm_nv_gpu_f16,
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

Op op_create_nv_gpu(Optype opty, void *) {
    switch (opty) {
        case OpRmsNorm: {
            auto op = new Operator{
                DevNvGpu,
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
