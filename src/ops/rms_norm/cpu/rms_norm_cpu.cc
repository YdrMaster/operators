#include "rms_norm_cpu.h"
#include <cmath>
#include "../../utils.h"
#include "../../../devices/cpu/common_cpu.h"

void rms_norm_cpu_f16(MutTensor y, ConstTensor x, ConstTensor w, float epsilon) {
    ASSERT_EQ(y.layout.ndim, 2);
    ASSERT_EQ(x.layout.ndim, 2);
    ASSERT_EQ(w.layout.ndim, 1);

    auto n = y.layout.shape[0],
         d = y.layout.shape[1];

    ASSERT_EQ(x.layout.shape[0], n);
    ASSERT_EQ(x.layout.shape[1], d);
    ASSERT_EQ(w.layout.shape[0], d);

    auto stride_y = y.layout.strides[0];
    auto stride_x = x.layout.strides[0];

    for (size_t i = 0; i < n; ++i) {
        auto y_ = reinterpret_cast<uint16_t *>(reinterpret_cast<char *>(y.data) + i * stride_y);
        auto x_ = reinterpret_cast<uint16_t const *>(reinterpret_cast<char const *>(x.data) + i * stride_x);
        auto w_ = reinterpret_cast<uint16_t const *>(w.data);

        auto sum_sq = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            auto x__ = f16_to_f32(x_[j]);
            sum_sq += x__ * x__;
        }

        auto k = std::pow(sum_sq / d + epsilon, -.5);
        for (size_t j = 0; j < d; ++j) {
            auto x__ = f16_to_f32(x_[j]);
            auto w__ = f16_to_f32(w_[j]);
            y_[j] = f32_to_f16(k * x__ * w__);
        }
    }
}
