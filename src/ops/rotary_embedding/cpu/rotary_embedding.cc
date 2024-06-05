#include "rotary_embedding.h"
#include "../../../utils.h"
#include <cmath>

void rotary_embedding_cpu_f16(struct Kernel const *kn, MutTensor t, ConstTensor pos, float theta) {
    ASSERT_EQ(t.layout.ndim, 3);
    ASSERT_EQ(pos.layout.ndim, 1);

    auto nt = t.layout.shape[0],
         nh = t.layout.shape[1],
         dh = t.layout.shape[2] / 2;

    ASSERT_EQ(pos.layout.shape[0], nt);

    auto stride_t = t.layout.strides[0];

    for (int i = 0; i < nt; ++i) {
        auto pos_ = reinterpret_cast<int32_t const *>(pos.data) + i;
        for (int j = 0; j < nh; ++j) {
            auto t_ = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(t.data) + i * stride_t + j * nh);
            for (int k = 0; k < dh / 2; ++k) {
                auto a = f16_to_f32(t_[2 * k]);
                auto b = f16_to_f32(t_[2 * k + 1]);
                auto pos__ = *pos_;
                float freq = pos__ / powf(theta, float(k) / float(dh));
                float sin = sinf(freq);
                float cos = cosf(freq);
                t_[2 * k] = f32_to_f16(a * cos - a * sin);
                t_[2 * k + 1] = f32_to_f16(b * cos + b * sin);
            }
        }
    }
}
