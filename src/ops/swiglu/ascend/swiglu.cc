#ifndef __ASCEND_NPU_SWIGLU_H__
#define __ASCEND_NPU_SWIGLU_H__

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include <cstdlib>
#endif

#include "../../../devices/ascend/common.h"
#include "../../utils.h"
#include "swiglu.h"

#define GM_ADDR __gm__ uint8_t *

#include "kernel_operator.h"
using namespace AscendC;

const int32_t BUFFER_NUM = 2;


template<typename T>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}
    __aicore__ inline void init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum,
                                uint32_t buffSize, float beta) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->beta = static_cast<float>(beta);
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // Get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ T *) x1 + this->blockLength * GetBlockIdx(), this->blockLength);
        x2Gm.setGlobalBuffer((__gm__ T *) x2 + this->blockLength * getBlockIdx(), this->blockLength);
        yGm.setGlobalBuffer((__gm__ T *) y + this->blockLength * getBlockIdx(), this->blockLength);
        // Pipe alloc memory to queue, the unit is bytes
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));

        if (sizeof(T) != sizeof(float)) {
            pipe.InitBuffer(calcBufs, buffSize);
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        // Alloc tensor from queue memory
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
        // Copy process_th tile from global tensor to local tensor
        DataCopy(x1Local, x1Gm[progress * this->tileLength], this->tileLength);
        DataCopy(x2Local, x2Gm[progress * this->tileLength], this->tileLength);
        // Enque input tensor to VECIN queue
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress) {
        // Alloc output tensor from VECOUT queue
        LocalTensor<T> yLocal = outQueue.AllocTensor<T>();
        // Deque input tensors from VECIN queue
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
        // Temp tensor
        LocalTensor<uint8_t> tmpLocal;
        if (sizeof(T) != sizeof(float)) {
            tmpLocal = calcBufs.Get<uint8_t>();
            SwiGLU<T, false>(yLocal, x1Local, x2Local, this->beta, tmpLocal, this->tileLength);
        } else {
            SwiGLU<T, false>(yLocal, x1Local, x2Local, this->beta, this->tileLength)
        }
        // Enque the output tensor to VECOUT queue
        outQueue.EnQue<T>(yLocal);
        // Free input Local tensors
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        // Deque output tensor from VECOUT queue
        LocalTensor<T> yLocal = outQueue.DeQue<T>();
        // Copy progress_th tile from local tensor to global tensor
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // Free output Local tensor
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBufs;
    // Create queue for inputs, depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    // Create queue for output
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // Create global tensor
    GlobalTensor<T> x1Gm, x2Gm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    // Beta value in swish fuction
    float beta = 0;
};

template<typename T>
extern "C" __global__ __aicore__ void swiglu(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                             float beta, SwiGLUTilingData &tiling) {
    KernelSwiGLU op;
    op.Init(x1, x2, y, tiling.totalLength, tiling.tileNum, tiling.maxBuffSize, (float) beta);
    op.Process();
}

class SwiGLUTilingData {
public:
    SwiGLUTilingData() = delete;
    SwiGLUTilingData(uint32_t totalLength, uint32_t tileNum)
        : totalLength(totalLength), tileNum(tileNum) {
        tileLength = totalLength / tileNum;
        uint32_t max;
        uint32_t min;
        std::vector<int64_t> tileShape = {tileLength};
        ge::Shape shape(tileShape);
        GetSwiGLUMaxMinTmpSize(shape, 2, max, min, false);
        maxBuffSize = max;
    }
    uint32_t getMaxBuffSize() {
        return maxBuffSize;
    }

public:
    uint32_t totalLength;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t maxBuffSize;
};

void swiglu_ascend_npu_fp16(MutTensor gate, ConstTensor up, void *stream) {
    // gate = swish(gate) * up
    auto gateShape = castToInt64_t(gate.layout.shape, gate.layout.ndim);
    auto upShape = castToInt64_t(up.layout.shape, up.layout.ndim);

    auto totalLength = shapeProd(gateShape, gate.layout.ndim);
    // TODO: Tune tileNum or get from Ascend API
    auto tileNum = 8;
    // TODO: Tune blockNum or get from Ascend API
    uint32_t blockNum = 8;

    SwiGLUTilingData tilingData(totalLength, tileNum);

    auto gateData = reinterpret_cast<GM_ADDR>(gate.data);
    auto upData = reinterpret_cast<GM_ADDR>(up.data);
    // Ascend SwiGLU kernel api do not support inplace calculate
    void *tempData;
    auto ret = aclrtMalloc(&tempData, totalLength * 2, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret));
    // Copy gate data into temp
    ret = aclrtMemcpy(tempData, totalLength * 2, (void *) gateData, totalLength * 2);
    swiglu<<<blockNum, nullptr, stream>>><float16>((GM_ADDR) tempData, (GM_ADDR) upData, (GM_ADDR) gateData, 1.0f, tilingData);
    
    return;
}

#endif
