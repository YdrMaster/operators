#include "swiglu_meta.h"

constexpr int32_t BLOCK_NUM = 8;
constexpr int32_t BUFFER_NUM = 1;

#include "kernel_operator.h"
using namespace AscendC;

template <typename T> class KernelSwiGLU {
  public:
    __aicore__ inline KernelSwiGLU() {}
    __aicore__ inline void Init(GM_ADDR x, int32_t strideX, GM_ADDR y,
                                int32_t strideY, uint32_t totalLength,
                                uint32_t tileLength, float beta) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->tileLength = tileLength;
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = this->blockLength / this->tileLength;
        ASSERT(this->tileNum != 0 && "tile num can not be zero!");
        ASSERT(this->tileNum % BUFFER_NUM == 0);
        this->strideX = strideX;
        this->strideY = strideY;
        this->beta = static_cast<float>(beta);

        // Get start index for current core, core parallel
        xGm.SetGlobalBuffer((__gm__ T *)x +
                                this->blockLength * GetBlockIdx() * strideX,
                            this->blockLength * strideX);
        yGm.SetGlobalBuffer((__gm__ T *)y +
                                this->blockLength * GetBlockIdx() * strideY,
                            this->blockLength * strideY);
        // Pipe alloc memory to queue, the unit is bytes
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        // Temp buffer used in SwiGLU
        if (sizeof(T) != sizeof(float)) {
            pipe.InitBuffer(calcBufs, this->tileLength *
                                          (sizeof(float) / sizeof(uint8_t)) *
                                          3);
        }
    }
    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int32_t progress) {
        // Alloc tensor from queue memory
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        // Copy process_th tile from global tensor to local tensor
        DataCopy(xLocal, xGm[progress * this->tileLength * this->strideX],
                 this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength * this->strideY],
                 this->tileLength);
        // Enque input tensor to VECIN queue
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        // Deque input tensors from VECIN queue
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        LocalTensor<uint8_t> tmpLocal;
        if (sizeof(T) != sizeof(float)) {
            tmpLocal = calcBufs.Get<uint8_t>();
            // outLocal = xLocal * swish(yLocal)
            SwiGLU<T, false>(outLocal, xLocal, yLocal, this->beta, tmpLocal,
                             this->tileLength);
        } else {
            SwiGLU<T, false>(outLocal, xLocal, yLocal, this->beta,
                             this->tileLength);
        }
        // Enque the output tensor to VECOUT queue
        outQueue.EnQue<T>(outLocal);
        // Free input Local tensors
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        // Deque output tensor from VECOUT queue
        LocalTensor<T> outLocal = outQueue.DeQue<T>();
        // Copy progress_th tile from local tensor to global tensor
        DataCopy(yGm[progress * this->tileLength * this->strideY], outLocal,
                 this->tileLength);
        // Free output Local tensor
        outQueue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> calcBufs;
    // Create queue for inputs, depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    // Create queue for output
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    // Create global tensor
    GlobalTensor<T> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    // Beta value in swish fuction
    float beta;
    int32_t strideX;
    int32_t strideY;
};

extern "C" __global__ __aicore__ void
swiglu_kernel_half(GM_ADDR x, int32_t strideX, GM_ADDR y, int32_t strideY,
                   int32_t totalLength, int32_t tileLength, float beta) {
    KernelSwiGLU<half> op;
    op.Init(x, strideX, y, strideY, totalLength, tileLength, beta);
    op.Process();
}

extern "C" __global__ __aicore__ void
swiglu_kernel_float(GM_ADDR x, int32_t strideX, GM_ADDR y, int32_t strideY,
                    int32_t totalLength, int32_t tileLength, float beta) {
    KernelSwiGLU<float> op;
    op.Init(x, strideX, y, strideY, totalLength, tileLength, beta);
    op.Process();
}

extern "C" void swiglu_kernel_do(void *x, void *y, SwiGLUMetaData meta_data,
                                 void *stream) {
    auto strideX = meta_data.strideX;
    auto strideY = meta_data.strideY;
    auto totalLength = meta_data.totalLen;
    auto tileLength = meta_data.tileLen;
    auto dtype = meta_data.dtype;
    auto beta = meta_data.beta;

    switch (dtype) {
    case 0:
        swiglu_kernel_float<<<BLOCK_NUM, nullptr, stream>>>(
            x, strideX, y, strideY, totalLength, tileLength, beta);
    case 1:
        swiglu_kernel_half<<<BLOCK_NUM, nullptr, stream>>>(
            x, strideX, y, strideY, totalLength, tileLength, beta);
    }
    return;
}