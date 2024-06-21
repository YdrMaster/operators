#include "../../../operators.h"
#include "../../utils.h"
#include "cnrt.h"

void rmsNorm_fp16(cnrtQueue_t queue, void *y, void const *x, void const *w, int n, int d, float eps);

void rms_norm_bang_f16(Tensor y, Tensor x, Tensor w, float epsilon, void *stream) {
    ASSERT_EQ(y.layout.ndim, 2);
    ASSERT_EQ(x.layout.ndim, 2);
    ASSERT_EQ(w.layout.ndim, 1);

    auto n = y.layout.shape[0],
         d = y.layout.shape[1];

    ASSERT_EQ(x.layout.shape[0], n);
    ASSERT_EQ(x.layout.shape[1], d);
    ASSERT_EQ(w.layout.shape[0], d);

    auto queue = reinterpret_cast<cnrtQueue_t>(stream);
    rmsNorm_fp16(queue, y.data, x.data, w.data, n, d, epsilon);
}