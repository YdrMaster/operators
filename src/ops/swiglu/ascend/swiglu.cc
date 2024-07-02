#include "swiglu.h"

#ifndef ASCEND_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_swiglu.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void swiglu(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, float beta, SwiGLUTilingData tiling)
#endif

#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif


void swiglu_ascend_npu_fp16(Tensor gate, Tensor up, void *stream) {
    // gate = swish(gate) * up
    auto gateShape = castToInt64_t(gate.layout.shape, gate.layout.ndim);
    auto upShape = castToInt64_t(up.layout.shape, up.layout.ndim);

    auto totalLength = shapeProd(gateShape);
    // TODO: Tune tileNum or get from Ascend API
    auto tileNum = 8;
    // TODO: Tune blockNum or get from Ascend API
    uint32_t blockNum = 8;

    // SwiGLUTilingData tilingData(totalLength, tileNum);

    auto gateData = reinterpret_cast<GM_ADDR>(gate.data);
    auto upData = reinterpret_cast<GM_ADDR>(up.data);
    // Ascend SwiGLU kernel api do not support inplace calculate
    void *tempData;
    auto ret = aclrtMalloc(&tempData, totalLength * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    // Copy gate data into temp
    ret = aclrtMemcpy(tempData, totalLength * 2, (void *) gateData, totalLength * 2, aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_HOST);
    // aclrtlaunch_swiglu(blockNum, stream, (GM_ADDR) tempData, (GM_ADDR) upData, (GM_ADDR) gateData, 1.0f, totalLength, tileNum);
    ACLRT_LAUNCH_KERNEL(swiglu)(blockNum, stream, tempData, upData, gateData, 1.0f, totalLength, tileNum);
    return;
}