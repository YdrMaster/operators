#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <stdlib.h>

#ifdef ENABLE_CAMBRICON_MLU
#include "../tensor.h"
#include "cnnl.h"
#include <vector>
#endif

/* This file contains some useful macros and helper functions */

// check if an expression is true, and if not, print an error message and abort the program
inline void assert_true(int expr, const char *msg, const char *file, int line) {
    if (!expr) {
        fprintf(stderr, "\033[31mAssertion failed:\033[0m %s at file %s, line %d\n", msg, file, line);
        exit(EXIT_FAILURE);
    }
}

#define ASSERT(expr) assert_true(expr, #expr " is false", __FILE__, __LINE__)
#define ASSERT_EQ(a, b) assert_true((a) == (b), #a " != " #b, __FILE__, __LINE__)
#define ASSERT_VALID_PTR(a) assert_true((a) != nullptr, #a " is nullptr", __FILE__, __LINE__)

#define PANIC(EXPR)                                             \
    printf("Error at %s:%d - %s\n", __FILE__, __LINE__, #EXPR); \
    exit(EXIT_FAILURE)

#define ROUND_UP_DIV(x, y) ((x + y - 1) / y)

#ifdef ENABLE_CAMBRICON_MLU
inline void setCnnlTensor(cnnlTensorDescriptor_t desc, TensorLayout layout) {
    std::vector<int> dims(layout.ndim);
    for (uint64_t i = 0; i < layout.ndim; i++) {
        dims[i] = static_cast<int>(layout.shape[i]);
    }
    cnnlSetTensorDescriptor(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                            dims.size(), dims.data());
}

inline std::pair<cnnlHandle_t, cnrtQueue_t> getCnnlHandle(void *stream) {
    cnnlHandle_t handle;
    cnrtQueue_t queue;
    if (stream != nullptr) {
        handle = reinterpret_cast<cnnlHandle_t>(stream);
        cnnlGetQueue(handle, &queue);
    } else {
        cnrtSetDevice(0);
        cnnlCreate(&handle);
        cnrtQueueCreate(&queue);
        cnnlSetQueue(handle, queue);
    }
    return std::make_pair(handle, queue);
}

#endif

#endif// __UTILS_H__
