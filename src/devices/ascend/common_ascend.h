#ifndef __ASCEND_NPU_COMMON_H__
#define __ASCEND_NPU_COMMON_H__

#include "acl/acl.h"
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

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

std::vector<int64_t> castToInt64_t(uint64_t *v, uint64_t size);
int64_t getShapeSize(const std::vector<int64_t> &shape);
const char *dataTypeToString(aclDataType dtype);
const char *formatToString(aclFormat format);

#endif
