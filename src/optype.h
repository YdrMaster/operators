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

struct Kernel;

typedef void (*RmsNormFn)(struct Kernel const *, MutTensor y, ConstTensor x, ConstTensor w, float epsilon);
typedef void (*MatMulFn)(struct Kernel const *, MutTensor c, float beta, ConstTensor a, ConstTensor b, float alpha);
typedef void (*RotaryEmbeddingFn)(struct Kernel const *, MutTensor, ConstTensor pos, float theta);
typedef void (*ReformFn)(struct Kernel const *, MutTensor dst, ConstTensor src);
typedef void (*CausalSoftmaxFn)(struct Kernel const *, MutTensor);
typedef void (*SwigluFn)(struct Kernel const *, MutTensor, ConstTensor);

#endif// __OPTYPE_H__
