#include "swiglu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void swiglu_cpu_f16(Tensor gate, Tensor up) {
    ASSERT_EQ(gate.layout->ndim, 2);
    ASSERT_EQ(up.layout->ndim, 2);
    ASSERT_EQ(gate.layout->shape[0], up.layout->shape[0]);
    ASSERT_EQ(gate.layout->shape[1], up.layout->shape[1]);

    auto seq_len = gate.layout->shape[0],
         di = gate.layout->shape[1];

    auto stride_gate = gate.layout->strides[0],
         stride_up = up.layout->strides[0];

    for (int i = 0; i < seq_len; ++i) {
        auto gate_ = reinterpret_cast<uint16_t *>(gate.data) + i * stride_gate;
        auto up_ = reinterpret_cast<uint16_t const *>(up.data) + i * stride_up;
        for (int j = 0; j < di; ++j) {
            auto x = f16_to_f32(gate_[j]);
            auto y = f16_to_f32(up_[j]);

            gate_[j] = f32_to_f16(x * sigmoid(x) * y);
        }
    }
}
