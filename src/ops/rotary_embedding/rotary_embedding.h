#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "../../operators.h"

#ifdef __cplusplus
extern "C" {
#endif

void *createRotaryEmbeddingDescriptor(Device, void *config);

void destroyRotaryEmbeddingDescriptor(void *descriptor);

void rotaryEmbedding(void *descriptor, MutTensor t, ConstTensor pos, float theta, void *stream);

#ifdef __cplusplus
}
#endif

typedef struct RotaryEmbeddingDescriptor {
    Device device;
} RotaryEmbeddingDescriptor;

#endif
