#ifndef __ASCEND_ROTARY_EMBEDDING_H__
#define __ASCEND_ROTARY_EMBEDDING_H__

#include "../../../operators.h"

struct RotaryEmbeddingAscendCDescriptor {
    Device device;
};

void rotary_embedding_ascendc_f16(Tensor t, Tensor pos, Tensor sin, Tensor cos,
                                  float theta, void *stream);

#endif