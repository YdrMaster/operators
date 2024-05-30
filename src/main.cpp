#include "ops/c_interface/cuda/nv_gpu.cuh"
#include "operators.h"
#include <cstdio>
#include <cuda_runtime.h>

void test_rms_norm() {
#ifdef ENABLE_NV_GPU
    auto op = op_create(DevNvGpu, OpRmsNorm, nullptr);
    printf("op: %p\n", op);
    struct NvGpuRtCtx ctx {
        0
    };
    auto kn = kn_load(op, &ctx);
    printf("kn: %p\n", kn);
#else
    auto op = op_create(DevCpu, OpRmsNorm, nullptr);
    printf("op: %p\n", op);
    auto kn = kn_load(op, nullptr);
    printf("kn: %p\n", kn);
#endif

    auto fn = reinterpret_cast<RmsNormFn>(fn_get(kn));
    printf("fn: %p\n", fn);

    fn(kn,
       MutTensor{TensorLayout{}, nullptr},
       ConstTensor{TensorLayout{}, nullptr},
       ConstTensor{TensorLayout{}, nullptr},
       1e-4);

    kn_unload(kn);
    op_destroy(op);
}

int main(int argc, char **argv) {
    test_rms_norm();
    return 0;
}
