#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdint.h>
#include "data_type.h"

struct TensorLayout {
    struct DataLayout dt;
    uint64_t ndim;
    uint64_t offset;
    uint64_t *shape; 
    int64_t *strides;
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
