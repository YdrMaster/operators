#ifndef __ASCEND_NPU_COMMON_H__
#define __ASCEND_NPU_COMMON_H__

// #include "acl/acl.h"
#include "../../tensor.h"
#include <cstdio>
#include <numeric>
#include <vector>
#include <functional>

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#ifdef __cplusplus
};
#endif

inline std::vector<int64_t> castToInt64_t(uint64_t *v, uint64_t size) {
    std::vector<int64_t> out(size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = static_cast<int64_t>(v[i]);
    }

    return out;
}

inline int64_t shapeProd(std::vector<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}


#endif
