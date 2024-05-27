#ifndef __OPTYPE_H__
#define __OPTYPE_H__

#include "tensor.h"

enum OptypeEnum {
    OpRmsNorm,
    OpMatMul,
    OpRotaryEmbedding,
    OpReform,
    OpCausalSoftmax,
    OpSwiglu,
};

typedef void (*RmsNormFn)(MutTensor y, ConstTensor x, ConstTensor w, float epsilon);
typedef void (*MatMulFn)(MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha);
typedef void (*RotaryEmbeddingFn)(MutTensor, ConstTensor pos, float theta);
typedef void (*ReformFn)(MutTensor dst, ConstTensor src);
typedef void (*CausalSoftmaxFn)(MutTensor);
typedef void (*SwigluFn)(MutTensor, ConstTensor);

#endif// __OPTYPE_H__
