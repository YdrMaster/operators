#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t MAX_TILE_SIZE = 32 * 8; // 8 datablocks

template <typename T> class RoPE {
  public:
    __aicore__ inline RoPE() {}
    // Init op
    // pos position vector
    // t input tensor
    // input tensor shape [nt, nh, dh]
    // make block_num = nh, tile_len = dh
    __aicore__ inline void Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                GM_ADDR cos, float theta, int32_t nt,
                                int32_t nh, int32_t dh);
    __aicore__ inline void Process();

  private:
    // Copy a tile into UB
    __aicore__ inline void CopyIn(int32_t i, int32_t tno);
    __aicore__ inline void Compute(int32_t i, int32_t tno);
    __aicore__ inline void CopyOut(int32_t i, int32_t tno);

  private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> sinQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> cosQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue;
    TBuf<TPosition::VECCALC> tmpOddBuf;
    TBuf<TPosition::VECCALC> tmpEvenBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> tmp2Buf;
    TBuf<TPosition::VECCALC> tmp3Buf;
    TBuf<TPosition::VECCALC> tmp4Buf;
    // TBuf<TPosition::VECCALC> tmpOffsetBuf;
    // if (sizeof(float) != sizeof(T)) {
    TBuf<TPosition::VECCALC> tmpSinBuf;
    TBuf<TPosition::VECCALC> tmpCosBuf;
    // }

    GlobalTensor<T> xGm;
    GlobalTensor<uint64_t> pGm;
    GlobalTensor<float> sinGm;
    GlobalTensor<float> cosGm;
    GlobalTensor<T> oGm;

    // TODO: Change to uint64_t
    uint32_t _block_idx;
    uint32_t block_len;
    uint32_t tile_len;
    uint32_t tile_num_i;
    uint32_t tile_num_j;

    // t[nt, nh, dh]
    // nt num of tokens
    // nh num of heads
    // dh dimension of each head
    int32_t nt;
    int32_t nh;
    int32_t dh;
    float theta;
};

template <typename T>
__aicore__ inline void RoPE<T>::Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                     GM_ADDR cos, float theta, int32_t nt,
                                     int32_t nh, int32_t dh) {
    this->nt = nt;
    this->nh = nh;
    this->dh = dh;
    this->theta = theta;

    _block_idx = GetBlockIdx();
    block_len = static_cast<uint32_t>(nt * dh);
    tile_len = MAX_TILE_SIZE / sizeof(T); // 128 of fp16, 64 of fp32
    tile_num_i = nt;
    // TODO: deal remaining segmentation
    tile_num_j = dh / tile_len;

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ T *)t);
    pGm.SetGlobalBuffer((__gm__ uint64_t *)pos);
    sinGm.SetGlobalBuffer((__gm__ float *)sin);
    cosGm.SetGlobalBuffer((__gm__ float *)cos);
    oGm.SetGlobalBuffer((__gm__ T *)t);

    // Init Queue buffer
    pipe.InitBuffer(inQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(outQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(sinQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(cosQue, BUFFER_NUM, tile_len * sizeof(T));
    pipe.InitBuffer(tmpOddBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpBuf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp2Buf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp3Buf, tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp4Buf, tile_len / 2 * sizeof(T));
    // pipe.InitBuffer(tmpOffsetBuf, tile_len / 2 * sizeof(uint32_t));
    // if (sizeof(float) != sizeof(T)) {
    pipe.InitBuffer(tmpSinBuf, tile_len * sizeof(float));
    pipe.InitBuffer(tmpCosBuf, tile_len * sizeof(float));
    // }

    // // Init Buffer
    // LocalTensor<uint32_t> offsetUb = tmpOffsetBuf.Get<uint32_t>();
    // for (uint32_t i = 0; i < tile_len / 2; ++i) {
    //     offsetUb(i) = i * sizeof(T) * 2;
    // }
}

template <typename T>
__aicore__ inline void RoPE<T>::CopyIn(int32_t i, int32_t tno) {
    LocalTensor<T> inputUb = inQue.AllocTensor<T>();
    LocalTensor<T> sinUb = sinQue.AllocTensor<T>();
    LocalTensor<T> cosUb = cosQue.AllocTensor<T>();
    // Get idx of current tile in total input
    auto idx = i * nh * dh + _block_idx * dh;
    // Copy tile current tile into UB
    DataCopy(inputUb, xGm[idx + tno * tile_len], tile_len);
    // Copy sin cos tile
    auto pos_idx = pGm(i);
    printf("%ld", pos_idx);
    // Cast sin cos to T type
    LocalTensor<float> tmpSinUb = tmpSinBuf.Get<float>();
    LocalTensor<float> tmpCosUb = tmpCosBuf.Get<float>();
    DataCopy(tmpSinUb, sinGm[pos_idx * dh + tno * tile_len], tile_len);
    DataCopy(tmpCosUb, cosGm[pos_idx * dh + tno * tile_len], tile_len);
    Cast<T, float>(sinUb, tmpSinUb, RoundMode::CAST_FLOOR, tile_len);
    Cast<T, float>(cosUb, tmpCosUb, RoundMode::CAST_FLOOR, tile_len);

    // if (i == 0 && _block_idx == 0) {
    //     DumpTensor(inputUb, 0, tile_len);
    //     DumpTensor(sinUb, 1, tile_len);
    //     DumpTensor(cosUb, 2, tile_len);
    // }

    // Push in operands
    inQue.EnQue(inputUb);
    sinQue.EnQue(sinUb);
    cosQue.EnQue(cosUb);
}

template <typename T>
__aicore__ inline void RoPE<T>::Compute(int32_t i, int32_t tno) {
    LocalTensor<T> inputUb = inQue.DeQue<T>();
    LocalTensor<T> sinUb = sinQue.DeQue<T>();
    LocalTensor<T> cosUb = cosQue.DeQue<T>();
    LocalTensor<T> outUb = outQue.AllocTensor<T>();

    // Choose odd and even position
    LocalTensor<T> tmpOdd = tmpOddBuf.Get<T>();
    LocalTensor<T> tmpEven = tmpEvenBuf.Get<T>();
    LocalTensor<T> tmpUb = tmpBuf.Get<T>();
    LocalTensor<T> tmp2Ub = tmp2Buf.Get<T>();
    LocalTensor<T> tmp3Ub = tmp3Buf.Get<T>();
    LocalTensor<T> tmp4Ub = tmp4Buf.Get<T>();

    uint64_t rsvdCnt = 0;
    // Select odd & even numbers
    GatherMask<T>(tmpOdd, inputUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmpEven, inputUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);

    // Calc odd position
    GatherMask<T>(tmpUb, cosUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmp2Ub, sinUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmpUb = tmpOdd * tmpUb;
    tmp2Ub = tmpEven * tmp2Ub;
    PipeBarrier<PIPE_V>();
    tmpUb = tmpUb - tmp2Ub;

    // if (i == 0 && _block_idx == 0) {
    //     DumpTensor(tmpUb, 6, tile_len / 2);
    // }
    // Scatter
    // LocalTensor<uint32_t> tmpOffsetUb = tmpOffsetBuf.Get<uint32_t>();
    // Scatter<T>(outUb, tmpUb, tmpOffsetUb, (uint32_t)0, tile_len / 2);

    // Calc even position
    GatherMask<T>(tmp3Ub, sinUb, 1, false, 0, {1, 1, 0, 0}, rsvdCnt);
    GatherMask<T>(tmp4Ub, cosUb, 2, false, 0, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmpOdd * tmp3Ub;
    tmp4Ub = tmpEven * tmp4Ub;
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmp3Ub + tmp4Ub;

    // if (i == 0 && _block_idx == 0) {
    //     DumpTensor(tmp3Ub, 7, tile_len / 2);
    // }

    // // Scatter
    // Scatter<T>(outUb, tmpUb, tmpOffsetUb, (uint32_t)sizeof(T), tile_len / 2);
    for (uint32_t i = 0; i < tile_len / 2; i += 1) {
        outUb(i * 2 + 1) = tmp3Ub(i);
        outUb(i * 2) = tmpUb(i);
    }
    if (i == 0 && _block_idx == 0) {
        DumpTensor(outUb, 11, tile_len);
    }

    outQue.EnQue<T>(outUb);
    inQue.FreeTensor(inputUb);
    sinQue.FreeTensor(sinUb);
    cosQue.FreeTensor(cosUb);
}

template <typename T>
__aicore__ inline void RoPE<T>::CopyOut(int32_t i, int32_t tno) {
    LocalTensor<T> outUb = outQue.DeQue<T>();
    auto idx = i * nh * dh + _block_idx * dh;
    DataCopy(oGm[idx + tno * tile_len], outUb, tile_len);
    outQue.FreeTensor(outUb);
}

template <typename T> __aicore__ inline void RoPE<T>::Process() {
    for (int32_t i = 0; i < nt; ++i) {
        for (int32_t j = 0; j < dh / tile_len; j++) {
            CopyIn(i, j);
            Compute(i, j);
            CopyOut(i, j);
        }
    }
}

// Kernel func
extern "C" __global__ __aicore__ void rope_kernel_fp16(GM_ADDR t, GM_ADDR pos,
                                                       GM_ADDR sin, GM_ADDR cos,
                                                       float theta, int32_t nt,
                                                       int32_t nh, int32_t dh) {
    RoPE<half> op;
    op.Init(t, pos, sin, cos, theta, nt, nh, dh);
    op.Process();
}

extern "C" void rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               float theta, int32_t nt, int32_t nh, int32_t dh,
                               int dtype, void *stream) {
    switch (dtype) {
    case 1: // ACL_FLOAT16
        rope_kernel_fp16<<<nh, nullptr, stream>>>(t, pos, sin, cos, theta, nt,
                                                  nh, dh);
        break;
    default:
        break;
    }
}
