#include "kernel_operator.h"
#include "../../../../include/status.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template<typename T> class RoPE {
public:
    __aicore__ inline RoPE() {}
    // Init op
    // pos position vector
    // t input tensor
    // input tensor shape [nt, nh, dh]
    // make block_num = nh, tile_len = dh
    __aicore__ inline void Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                GM_ADDR cos, int32_t nt, int32_t nh,
                                int32_t dh, int32_t stt, int32_t sth);
    __aicore__ inline void Process();

private:
    // Copy a tile into UB
    __aicore__ inline void CopyIn(int32_t i);
    __aicore__ inline void Compute(int32_t i);
    __aicore__ inline void CopyOut(int32_t i);

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
    TBuf<TPosition::VECCALC> tmpSinBuf;
    TBuf<TPosition::VECCALC> tmpCosBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<uint64_t> pGm;
    GlobalTensor<float> sinGm;
    GlobalTensor<float> cosGm;
    GlobalTensor<T> oGm;

    // TODO: Change to uint64_t
    uint32_t _block_idx;
    uint32_t _tile_len;

    // t[nt, nh, dh]
    // nt num of tokens
    // nh num of heads
    // dh dimension of each head
    int32_t nt;
    int32_t nh;
    int32_t dh;
    int32_t sth;
    int32_t stt;
};

template<typename T>
__aicore__ inline void RoPE<T>::Init(GM_ADDR t, GM_ADDR pos, GM_ADDR sin,
                                     GM_ADDR cos, int32_t nt, int32_t nh,
                                     int32_t dh, int32_t stt, int32_t sth) {
    this->nt = nt;
    this->nh = nh;
    this->dh = dh;
    this->stt = stt;
    this->sth = sth;

    _block_idx = GetBlockIdx();
    _tile_len = dh;

    // Init global buffer
    xGm.SetGlobalBuffer((__gm__ T *) t);
    pGm.SetGlobalBuffer((__gm__ uint64_t *) pos);
    sinGm.SetGlobalBuffer((__gm__ float *) sin);
    cosGm.SetGlobalBuffer((__gm__ float *) cos);
    oGm.SetGlobalBuffer((__gm__ T *) t);

    // Init Queue buffer
    pipe.InitBuffer(inQue, BUFFER_NUM, _tile_len * sizeof(T));
    pipe.InitBuffer(outQue, BUFFER_NUM, _tile_len * sizeof(T));
    pipe.InitBuffer(sinQue, BUFFER_NUM, _tile_len * sizeof(float));
    pipe.InitBuffer(cosQue, BUFFER_NUM, _tile_len * sizeof(float));
    pipe.InitBuffer(tmpOddBuf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpEvenBuf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpBuf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp2Buf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp3Buf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmp4Buf, _tile_len / 2 * sizeof(T));
    pipe.InitBuffer(tmpSinBuf, _tile_len * sizeof(T));
    pipe.InitBuffer(tmpCosBuf, _tile_len * sizeof(T));
}

template<typename T>
__aicore__ inline void RoPE<T>::CopyIn(int32_t i) {
    LocalTensor<T> inputUb = inQue.AllocTensor<T>();
    LocalTensor<float> sinUb = sinQue.AllocTensor<float>();
    LocalTensor<float> cosUb = cosQue.AllocTensor<float>();
    // Get idx of current tile in total input
    auto idx = i * stt + _block_idx * sth;
    // Copy tile current tile into UB
    DataCopy(inputUb, xGm[idx], _tile_len);
    // Copy sin cos tile
    auto pos_idx = pGm(i);
    // Cast sin cos to T type
    DataCopy(sinUb, sinGm[pos_idx * dh], _tile_len);
    DataCopy(cosUb, cosGm[pos_idx * dh], _tile_len);
    // Push in operands
    inQue.EnQue(inputUb);
    sinQue.EnQue(sinUb);
    cosQue.EnQue(cosUb);
}

template<typename T>
__aicore__ inline void RoPE<T>::Compute(int32_t i) {
    LocalTensor<T> inputUb = inQue.DeQue<T>();
    LocalTensor<float> sinUb = sinQue.DeQue<float>();
    LocalTensor<float> cosUb = cosQue.DeQue<float>();
    LocalTensor<T> outUb = outQue.AllocTensor<T>();

    // Choose odd and even position
    LocalTensor<T> tmpOdd = tmpOddBuf.Get<T>();
    LocalTensor<T> tmpEven = tmpEvenBuf.Get<T>();
    LocalTensor<T> tmpUb = tmpBuf.Get<T>();
    LocalTensor<T> tmp2Ub = tmp2Buf.Get<T>();
    LocalTensor<T> tmp3Ub = tmp3Buf.Get<T>();
    LocalTensor<T> tmp4Ub = tmp4Buf.Get<T>();
    LocalTensor<T> tmpSinUb = tmpSinBuf.Get<T>();
    LocalTensor<T> tmpCosUb = tmpCosBuf.Get<T>();

    // Cast from float to T
    Cast<T, float>(tmpSinUb, sinUb, RoundMode::CAST_FLOOR, _tile_len);
    Cast<T, float>(tmpCosUb, cosUb, RoundMode::CAST_FLOOR, _tile_len);
    PipeBarrier<PIPE_V>();
    
    // Select odd & even numbers
    uint64_t rsvdCnt = 0;
    GatherMaskParams gMaskParams = {
        1,
        static_cast<uint16_t>((_tile_len * sizeof(T) + 255) / 256),
        8,
        8,
    };
    GatherMask<T>(tmpOdd, inputUb, 1, false, 0, gMaskParams, rsvdCnt);
    GatherMask<T>(tmpEven, inputUb, 2, false, 0, gMaskParams, rsvdCnt);
    
    // Calc odd position
    GatherMask<T>(tmpUb, tmpCosUb, 1, false, 0, gMaskParams, rsvdCnt);
    GatherMask<T>(tmp2Ub, tmpSinUb, 1, false, 0, gMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmpUb = tmpOdd * tmpUb;
    tmp2Ub = tmpEven * tmp2Ub;
    PipeBarrier<PIPE_V>();
    tmpUb = tmpUb - tmp2Ub;

    // Calc even position
    GatherMask<T>(tmp3Ub, tmpSinUb, 2, false, 0, gMaskParams, rsvdCnt);
    GatherMask<T>(tmp4Ub, tmpCosUb, 2, false, 0, gMaskParams, rsvdCnt);
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmpOdd * tmp3Ub;
    tmp4Ub = tmpEven * tmp4Ub;
    PipeBarrier<PIPE_V>();
    tmp3Ub = tmp3Ub + tmp4Ub;

    // Scatter
    // Scatter<T>(outUb, tmpUb, tmpOffsetUb, (uint32_t)sizeof(T), tile_len / 2);
    for (uint32_t i = 0; i < _tile_len / 2; i += 1) {
        outUb(i * 2 + 1) = tmp3Ub(i);
        outUb(i * 2) = tmpUb(i);
    }

    outQue.EnQue<T>(outUb);
    inQue.FreeTensor(inputUb);
    sinQue.FreeTensor(sinUb);
    cosQue.FreeTensor(cosUb);
}

template<typename T>
__aicore__ inline void RoPE<T>::CopyOut(int32_t i) {
    LocalTensor<T> outUb = outQue.DeQue<T>();
    auto idx = i * stt + _block_idx * sth;
    // DataCopy(oGm[idx], outUb, _tile_len);
    DataCopyExtParams dcep = {
        1,
        static_cast<uint32_t>(_tile_len * sizeof(T)),
        0, 0, 0};
    DataCopyPad(oGm[idx], outUb, dcep);
    outQue.FreeTensor(outUb);
}

template<typename T> __aicore__ inline void RoPE<T>::Process() {

    for (int32_t i = 0; i < nt; ++i) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

// Kernel func
__global__ __aicore__ void rope_kernel_fp16(GM_ADDR t, GM_ADDR pos,
                                                       GM_ADDR sin, GM_ADDR cos,
                                                       int32_t nt, int32_t nh,
                                                       int32_t dh, int32_t stt,
                                                       int32_t sth) {
    RoPE<half> op;
    op.Init(t, pos, sin, cos, nt, nh, dh, stt, sth);
    op.Process();
}

extern "C"  infiniopStatus_t rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               int32_t nt, int32_t nh, int32_t dh,
                               int32_t stt, int32_t sth,
                               int dtype, void *stream) {
    switch (dtype) {
        case 0:// ACL_FLOAT32
            // TODO:
            break;
        case 1:// ACL_FLOAT16
            rope_kernel_fp16<<<nh, nullptr, stream>>>(t, pos, sin, cos, nt, nh, dh, stt, sth);
            return STATUS_SUCCESS;
        default:
            break;
    }
    return STATUS_BAD_TENSOR_DTYPE;
}
