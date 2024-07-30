#include "kernel_operator.h"
using namespace AscendC;

const int32_t BUFFER_NUM = 2;

typedef half float16;

class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum,
                                float beta) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        this->beta = static_cast<float>(beta);
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // Get start index for current core, core parallel
        x1Gm.SetGlobalBuffer((__gm__ half *) x1 + this->blockLength * GetBlockIdx(), this->blockLength);
        x2Gm.SetGlobalBuffer((__gm__ half *) x2 + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *) y + this->blockLength * GetBlockIdx(), this->blockLength);
        // Pipe alloc memory to queue, the unit is bytes
        pipe.InitBuffer(inQueueX1, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueX2, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(half));

        // if (sizeof(T) != sizeof(float)) {
        //     pipe.InitBuffer(calcBufs, buffSize);
        // }
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
        LocalTensor<half> x1Local = inQueueX1.AllocTensor<half>();
        LocalTensor<half> x2Local = inQueueX2.AllocTensor<half>();
        // Copy process_th tile from global tensor to local tensor
        DataCopy(x1Local, x1Gm[progress * this->tileLength], this->tileLength);
        DataCopy(x2Local, x2Gm[progress * this->tileLength], this->tileLength);
        // Enque input tensor to VECIN queue
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }
    __aicore__ inline void Compute(int32_t progress) {
        // Alloc output tensor from VECOUT queue
        LocalTensor<half> yLocal = outQueue.AllocTensor<half>();
        // Deque input tensors from VECIN queue
        LocalTensor<half> x1Local = inQueueX1.DeQue<half>();
        LocalTensor<half> x2Local = inQueueX2.DeQue<half>();
        // Temp tensor
        LocalTensor<uint8_t> tmpLocal;
        // if (sizeof(T) != sizeof(float)) {
        //     tmpLocal = calcBufs.Get<uint8_t>();
        //     SwiGLU<T, false>(yLocal, x1Local, x2Local, this->beta, tmpLocal, this->tileLength);
        // } else {
        SwiGLU<half, false>(yLocal, x1Local, x2Local, this->beta, this->tileLength);
        // }
        // Enque the output tensor to VECOUT queue
        outQueue.EnQue<half>(yLocal);
        // Free input Local tensors
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        // Deque output tensor from VECOUT queue
        LocalTensor<half> yLocal = outQueue.DeQue<half>();
        // Copy progress_th tile from local tensor to global tensor
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        // Free output Local tensor
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    // TBuf<TPosition::VECCALC> calcBufs;
    // Create queue for inputs, depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1, inQueueX2;
    // Create queue for output
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // Create global tensor
    GlobalTensor<half> x1Gm, x2Gm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    // Beta value in swish fuction
    float beta = 0;
};

__global__ __aicore__ void swiglu(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                  float beta, uint32_t totalLength, uint32_t tileNum) {
    KernelSwiGLU op;
    op.Init(x1, x2, y, totalLength, tileNum, (float) beta);
    op.Process();
}
