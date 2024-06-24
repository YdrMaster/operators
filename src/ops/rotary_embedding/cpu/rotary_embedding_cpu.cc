#include "rotary_embedding_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <cmath>

void rotary_embedding_cpu_f16(Tensor t, Tensor pos, float theta) {
    ASSERT_EQ(t.layout.ndim, 3);
    ASSERT_EQ(pos.layout.ndim, 1);

    auto nt = t.layout.shape[0],
         nh = t.layout.shape[1],
         dh = t.layout.shape[2] / 2;

    ASSERT_EQ(pos.layout.shape[0], nt);

    auto stride_0 = t.layout.strides[0];
    auto stride_1 = t.layout.strides[1];

    for (int i = 0; i < nt; ++i) {
        auto pos_ = reinterpret_cast<unsigned int const *>(pos.data) + i;
        for (int j = 0; j < nh; ++j) {
            auto t_ = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(t.data) + i * stride_0 + j * stride_1);
            for (int k = 0; k < dh; ++k) {
                auto a = f16_to_f32(t_[2 * k]);
                auto b = f16_to_f32(t_[2 * k + 1]);
                auto pos__ = *pos_;
                float freq = float(pos__) / powf(theta, float(k) / float(dh));
                float sin = sinf(freq);
                float cos = cosf(freq);
                t_[2 * k] = f32_to_f16(a * cos - b * sin);
                t_[2 * k + 1] = f32_to_f16(a * sin + b * cos);
            }
        }
    }
}
