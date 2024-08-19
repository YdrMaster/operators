#include "causal_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../utils.h"
#include <algorithm>

void causal_softmax_cpu_f16(Tensor y) {
    uint64_t ndim = y.layout->ndim;
    ASSERT(ndim == 2 || ndim == 3);
    uint64_t total_seq_len = y.layout->shape[ndim - 1];
    uint64_t seq_len = y.layout->shape[ndim - 2];
    uint64_t batch_size = 1;
    uint64_t stride_j = y.layout->strides[ndim - 1] / 2;
    uint64_t stride_i = y.layout->strides[ndim - 2] / 2;
    uint64_t stride_b = 0;
    if (ndim == 3)
        stride_b = y.layout->strides[ndim - 3] / 2;
    for (size_t i = 0; i < ndim - 2; i++) {
        batch_size *= y.layout->shape[i];
    }
    auto y_ptr = reinterpret_cast<uint16_t *>(y.data);
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            uint64_t offset = b * stride_b + i * stride_i;
            float max_val = f16_to_f32(y_ptr[offset]);
            for (size_t j = 1; j < total_seq_len; j++) {
                if (j <= total_seq_len - seq_len + i) {
                    max_val = std::max(max_val, f16_to_f32(y_ptr[offset + j * stride_j]));
                } else {
                    y_ptr[offset + j * stride_j] = 0;
                }
            }
            float sum = 0.;
            for (size_t j = 0; j <= total_seq_len - seq_len + i; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j * stride_j]) - max_val);
                sum += new_val;
            }
            for (size_t j = 0; j <= total_seq_len - seq_len + i; j++) {
                float new_val = std::exp(f16_to_f32(y_ptr[offset + j * stride_j]) - max_val) / sum;
                y_ptr[offset + j * stride_j] = f32_to_f16(new_val);
            }
        }
    }
}
