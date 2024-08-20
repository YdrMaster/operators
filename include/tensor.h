#ifndef __TENSOR_H__
#define __TENSOR_H__

#include "data_type.h"
#include <stdint.h>

struct TensorDescriptor {
    // Datatype
    DT dt;
    // Number of dimensions
    uint64_t ndim;
    // Shape of the tensor, ndim elements
    uint64_t *shape;
    // Stride of each dimension IN BYTES, ndim elements
    int64_t *strides;
};

typedef struct TensorDescriptor *infiniopTensorDescriptor_t;

// @depricated
struct TensorTuple {
    infiniopTensorDescriptor_t const layout;
    void *data;
};
// @depricated
typedef struct TensorTuple Tensor;

#endif// __TENSOR_H__
