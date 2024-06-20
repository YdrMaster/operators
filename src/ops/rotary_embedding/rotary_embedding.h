#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "../../export.h"
#include "../../operators.h"

__C __export void *createRotaryEmbeddingDescriptor(Device, void *config);
__C __export void destroyRotaryEmbeddingDescriptor(void *descriptor);
__C __export void rotaryEmbedding(void *descriptor, Tensor t, Tensor pos, float theta, void *stream);

typedef struct RotaryEmbeddingDescriptor {
    Device device;
} RotaryEmbeddingDescriptor;

#endif
