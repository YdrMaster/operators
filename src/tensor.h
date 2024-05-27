#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>

struct TensorLayout {
    uint64_t ndim;
    uint64_t *shape; // [num subtensors of dim      ; ndim], bytes of item
    int64_t *pattern;// [num bytes to next subtensor; ndim], num bytes to first item
};

struct ConstTensorTuple {
    struct TensorLayout const layout;
    void const *data;
};

struct MutableTensorTuple {
    struct TensorLayout const layout;
    void *data;
};

typedef struct ConstTensorTuple ConstTensor;
typedef struct MutableTensorTuple MutTensor;

#endif// __TENSOR_H__
