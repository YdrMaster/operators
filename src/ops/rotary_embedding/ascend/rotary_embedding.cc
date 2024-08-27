#include "rotary_embedding.h"
#include "../../../devices/ascend/common_ascend.h"
#include "../../utils.h"
#include "acl/acl.h"

extern "C" void rope_kernel_do(void *t, void *pos, void *sin, void *cos,
                               float theta, int32_t nt, int32_t nh, int32_t dh,
                               int dtype, void *stream);

void rotary_embedding_ascendc_f16(Tensor t, Tensor pos, Tensor sin, Tensor cos,
                                  float theta, void *stream) {
    auto nt = static_cast<int>(t.layout->shape[0]);
    auto nh = static_cast<int>(t.layout->shape[1]);
    auto dh = static_cast<int>(t.layout->shape[2]);

    ASSERT_EQ(pos.layout->shape[0], t.layout->shape[0]);

    rope_kernel_do(t.data, pos.data, sin.data, cos.data, theta, nt, nh, dh,
                   aclDataType::ACL_FLOAT16, stream);
}
