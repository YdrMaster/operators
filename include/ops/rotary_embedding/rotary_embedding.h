#ifndef ROTARY_EMBEDDING_H
#define ROTARY_EMBEDDING_H

#include "../../export.h"
#include "../../operators.h"

typedef struct RotaryEmbeddingDescriptor RotaryEmbeddingDescriptor;
typedef RotaryEmbeddingDescriptor* infiniopRoPEDescriptor_t;

// @deprecated
__C __export void *createRotaryEmbeddingDescriptor(Device, void *config);
// @deprecated
__C __export void destroyRotaryEmbeddingDescriptor(RotaryEmbeddingDescriptor *descriptor);
// @deprecated
__C __export void rotaryEmbedding(RotaryEmbeddingDescriptor *descriptor, Tensor t, Tensor pos, float theta, void *stream);

#endif
