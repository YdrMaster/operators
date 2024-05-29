#ifndef __NV_GPU_CUH__
#define __NV_GPU_CUH__

#include "../internal.h"
#include <cuda_runtime.h>

struct NvGpuRtCtx {
    cudaStream_t stream;
};

#ifdef __cplusplus
extern "C" {
#endif

Op op_create_nv_gpu(Optype, void *config);

#ifdef __cplusplus
}
#endif

#endif// __NV_GPU_CUH__
